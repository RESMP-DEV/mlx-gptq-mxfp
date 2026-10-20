import argparse

import pytest
import torch
from mlx_gptq import dflash2_distill
from mlx_gptq.dflash2_distill import (
    LoRALinear,
    _capture_order,
    _cycle_loss,
    _distribution_loss,
    _evaluate_cycles,
    _flat_metrics,
    _install_lora,
    _merge_lora,
    _mxfp4_parameters,
    _prompt_split,
    _remove_lora,
    _selector_loss,
    _stratified_indices,
    _use_captured_hidden,
)
from mlx_gptq.grid import dequantize_mxfp, rtn_quantize_mxfp
from mlx_gptq.qat import fake_quantize_mxfp4


class _Selector:
    def __init__(self):
        self.top_k = 2
        self.predecessor_codebook = torch.nn.Embedding(10, 4)
        self.successor_codebook = torch.nn.Embedding(10, 4)
        self.hidden_projection = torch.nn.Linear(4, 4, bias=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate_selector = _Selector()


def _payload():
    return {
        "anchor": torch.tensor([3], dtype=torch.long),
        "target_top_ids": torch.tensor([[0, 1, 2], [7, 8, 9], [7, 8, 9]]),
        "target_top_values": torch.tensor(
            [[1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            dtype=torch.float32,
        ),
        "target_greedy": torch.tensor([0, 7, 8]),
    }


def test_selector_loss_includes_target_unary_scores():
    model = _Model()
    payload = _payload()
    hidden = torch.zeros(1, 2, 4)
    weak_teacher = _selector_loss(model, payload, hidden, "cpu")
    sequence_loss = _selector_loss(model, payload, hidden, "cpu", reduction="sum")

    payload["target_top_values"][-1] = torch.tensor([-8.0, 8.0, 0.0])
    strong_teacher = _selector_loss(model, payload, hidden, "cpu")

    assert torch.isfinite(weak_teacher)
    assert torch.isfinite(sequence_loss)
    assert sequence_loss.detach().item() > weak_teacher.detach().item()
    assert torch.isfinite(strong_teacher)
    assert strong_teacher.detach().item() < weak_teacher.detach().item()


def test_selector_loss_rejects_missing_teacher():
    model = _Model()
    payload = _payload()
    payload["target_greedy"][-1] = 4
    hidden = torch.zeros(1, 2, 4)

    with pytest.raises(RuntimeError, match="absent from its selector set"):
        _selector_loss(model, payload, hidden, "cpu")


def test_selector_loss_reports_runtime_aligned_metrics():
    torch.manual_seed(7)
    model = _Model()
    payload = _payload()
    hidden = torch.zeros(1, 2, 4)

    loss, metrics = _selector_loss(
        model,
        payload,
        hidden,
        "cpu",
        predecessor_mode="predicted",
        return_metrics=True,
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "top1_accuracy", "exact_match", "prefix_accuracy"}
    for value in metrics.values():
        assert torch.isfinite(value)
    assert 0.0 <= float(metrics["top1_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["exact_match"]) <= 1.0
    assert 0.0 <= float(metrics["prefix_accuracy"]) <= 1.0


def test_stratified_limit_keeps_prompt_coverage_and_counts():
    prompt_ids = [0] * 8 + [1] * 4 + [2] * 2 + [3] * 2
    selected = _stratified_indices(prompt_ids, limit=8, seed=17)

    assert len(selected) == 8
    assert len(selected) == len(set(selected))
    assert {prompt_ids[index] for index in selected} == {0, 1, 2, 3}
    counts = {prompt_id: 0 for prompt_id in range(4)}
    for index in selected:
        counts[prompt_ids[index]] += 1
    assert counts[0] > counts[1]
    assert counts[1] >= counts[2]


def test_flat_metrics_names_prompt_and_global_values():
    metrics = {
        "loss": 1.5,
        "by_prompt": {"6": {"prefix_accuracy": 0.75, "loss": 1.25}},
    }

    assert _flat_metrics("validation", metrics) == {
        "validation/loss": 1.5,
        "validation/by_prompt/6/prefix_accuracy": 0.75,
        "validation/by_prompt/6/loss": 1.25,
    }


def test_lora_linear_starts_equivalent_and_receives_gradients():
    base = torch.nn.Linear(4, 3, bias=False)
    module = LoRALinear(base, rank=2, alpha=4.0)
    input = torch.randn(2, 4, requires_grad=True)

    expected = base(input)
    actual = module(input)
    actual.sum().backward()

    assert torch.allclose(actual, expected)
    assert module.a.grad is not None
    assert module.b.grad is not None
    assert input.grad is not None


def test_qat_lora_uses_mxfp_grid_and_passes_gradients():
    torch.manual_seed(11)
    base = torch.nn.Linear(64, 32, bias=False)
    module = LoRALinear(base, rank=2, alpha=4.0, qat_mxfp4=True)
    with torch.no_grad():
        module.b[:, 0] = 0.125
    input = torch.randn(2, 64, requires_grad=True)

    output = module(input)
    output.sum().backward()

    effective = base.weight + (module.b @ module.a) * module.scaling
    _codes, _scales, _biases, expected = rtn_quantize_mxfp(
        effective.detach().float(), "mxfp4"
    )
    torch.testing.assert_close(
        module(input).detach(),
        torch.nn.functional.linear(input.detach(), expected.to(base.weight.dtype)),
        atol=2e-2,
        rtol=2e-2,
    )
    assert module.a.grad is not None
    assert module.b.grad is not None
    assert torch.isfinite(module.a.grad).all()
    assert torch.isfinite(module.b.grad).all()


def test_qat_merge_exports_fake_quantized_weights():
    torch.manual_seed(31)
    base = torch.nn.Linear(64, 32, bias=False).to(torch.bfloat16)
    module = LoRALinear(base, rank=2, alpha=4.0, qat_mxfp4=True)
    with torch.no_grad():
        module.b[:, 0] = 0.125
    input = torch.randn(2, 64, dtype=torch.bfloat16)

    trained = module(input).to(torch.bfloat16)
    _merge_lora(module)
    exported = torch.nn.functional.linear(input, module.base.weight)

    torch.testing.assert_close(trained, exported)


def test_mxfp4_parameters_round_trip_is_exact_after_cast():
    """Prescribed codes and scales must reproduce the saved BF16 weight."""
    torch.manual_seed(41)
    weight = torch.randn(32, 64).to(torch.bfloat16)
    codes, scales, effective = _mxfp4_parameters(weight)
    saved = effective.to(torch.bfloat16)
    restored = dequantize_mxfp(codes, scales, "mxfp4").reshape(saved.shape)

    torch.testing.assert_close(saved, restored.to(torch.bfloat16), rtol=0, atol=0)


def test_fake_quantize_mxfp4_straight_through_gradient():
    weight = torch.randn(4, 64, requires_grad=True)
    quantized = fake_quantize_mxfp4(weight)
    quantized.sum().backward()

    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_distribution_loss_uses_aligned_full_vocab_logits():
    torch.manual_seed(17)
    hidden = torch.randn(1, 2, 4)
    lm_head = torch.randn(6, 4)
    target_logits = torch.randn(3, 6)
    payload = {
        "target_logits": target_logits,
        "target_greedy": torch.tensor([0, 1, 2]),
    }

    loss, metrics = _distribution_loss(payload, hidden, "cpu", lm_head)

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss",
        "distribution_loss",
        "distribution_ce",
        "distribution_tv",
    }
    assert torch.isfinite(metrics["distribution_ce"])
    assert torch.isfinite(metrics["distribution_tv"])


def test_distribution_loss_weights_ce_and_tv():
    torch.manual_seed(23)
    hidden = torch.randn(1, 2, 4)
    lm_head = torch.randn(6, 4)
    payload = {
        "target_logits": torch.randn(3, 6),
        "target_greedy": torch.tensor([0, 1, 2]),
    }

    loss, metrics = _distribution_loss(
        payload, hidden, "cpu", lm_head, ce_weight=0.1, tv_weight=0.9
    )

    expected = 0.1 * metrics["distribution_ce"] + 0.9 * metrics["distribution_tv"]
    torch.testing.assert_close(loss, expected)


def test_distribution_loss_honors_sum_reduction():
    torch.manual_seed(29)
    hidden = torch.randn(1, 2, 4)
    lm_head = torch.randn(6, 4)
    payload = {
        "target_logits": torch.randn(3, 6),
        "target_greedy": torch.tensor([0, 1, 2]),
    }

    mean_loss, mean_metrics = _distribution_loss(
        payload, hidden, "cpu", lm_head, reduction="mean"
    )
    sum_loss, sum_metrics = _distribution_loss(
        payload, hidden, "cpu", lm_head, reduction="sum"
    )

    assert sum_metrics["distribution_ce"] == 2 * mean_metrics["distribution_ce"]
    assert sum_metrics["distribution_tv"] == 2 * mean_metrics["distribution_tv"]
    assert sum_loss == 2 * mean_loss


def test_hybrid_cycle_loss_combines_selector_and_distribution_terms():
    torch.manual_seed(19)
    model = _Model()
    payload = _payload()
    payload["target_logits"] = torch.randn(3, 10)
    hidden = torch.randn(1, 2, 4)
    lm_head = torch.randn(10, 4)

    loss, metrics = _cycle_loss(
        model,
        payload,
        hidden,
        "cpu",
        loss_mode="hybrid",
        predecessor_mode="teacher",
        lm_head=lm_head,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == loss
    assert "prefix_accuracy" in metrics
    assert "distribution_tv" in metrics


def test_evaluate_cycles_accumulates_first_metric():
    class Dataset:
        prompt_ids = [0]

        def __getitem__(self, index: int):
            payload = _payload()
            payload["selector_hidden"] = torch.zeros(1, 2, 4)
            payload["target_logits"] = torch.randn(3, 10)
            return payload, 0

    metrics = _evaluate_cycles(
        Dataset(),
        [0],
        _Model(),
        "cpu",
        reduction="mean",
        predecessor_mode="teacher",
        loss_mode="selector",
        lm_head=None,
    )

    assert set(metrics) == {
        "loss",
        "top1_accuracy",
        "exact_match",
        "prefix_accuracy",
        "by_prompt",
    }


def test_install_lora_wraps_draft_linears_but_not_selector_projection():
    class Selector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_projection = torch.nn.Linear(4, 2, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(6, 4, bias=False)
            self.layers = torch.nn.ModuleList([Layer()])
            self.candidate_selector = Selector()

    model = Model()
    modules, parameters = _install_lora(model, rank=2, alpha=4.0)

    assert modules == 2
    assert parameters == 2 * (6 + 4) + 2 * (4 + 4)
    assert isinstance(model.fc, LoRALinear)
    assert isinstance(model.layers[0].q_proj, LoRALinear)
    assert not isinstance(model.candidate_selector.hidden_projection, LoRALinear)


def test_prompt_split_never_leaks_a_prompt():
    prompt_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    train, validation, validation_prompts = _prompt_split(
        prompt_ids, fraction=0.25, seed=17
    )

    train_prompts = {prompt_ids[index] for index in train}
    assert train_prompts.isdisjoint(validation_prompts)
    assert set(validation_prompts) == {prompt_ids[index] for index in validation}
    assert len(train) + len(validation) == len(prompt_ids)


def test_capture_order_keeps_each_prompt_contiguous():
    prompt_ids = [7, 7, 8, 8, 9]

    order = _capture_order([4, 0, 2, 1, 3], prompt_ids)

    assert order == [0, 1, 2, 3, 4]
    assert [prompt_ids[index] for index in order] == [7, 7, 8, 8, 9]


def test_external_validation_capture_must_be_prompt_disjoint(monkeypatch):
    class Dataset:
        ids_by_root = {"train": [0], "validation": [1]}

        def __init__(self, root, limit=0, stratified=False, seed=0):
            self.root = str(root)
            self.prompt_ids = list(self.ids_by_root[self.root])

        def __len__(self):
            return len(self.prompt_ids)

    monkeypatch.setattr(dflash2_distill, "CycleDataset", Dataset)
    args = argparse.Namespace(
        capture_dir="train",
        validation_capture_dir="validation",
        validation_fraction=None,
        limit=0,
        stratified_limit=False,
        seed=17,
    )

    split = dflash2_distill._datasets(args)
    train, validation, ids, validation_ids, prompts, fraction = split

    assert train is not validation
    assert ids == [0]
    assert validation_ids == [0]
    assert prompts == {1}
    assert fraction is None

    args.validation_fraction = 0.2
    with pytest.raises(ValueError, match="either"):
        dflash2_distill._datasets(args)
    args.validation_fraction = None
    Dataset.ids_by_root["validation"] = [0]
    with pytest.raises(ValueError, match="leaks"):
        dflash2_distill._datasets(args)


def test_merge_and_remove_lora_exports_original_keys():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 3, bias=False)

    model = Model()
    expected_keys = set(model.state_dict())
    _install_lora(model, rank=2, alpha=4.0)
    with torch.no_grad():
        model.fc.b[:, 0] = 0.125
    _merge_lora(model)
    _remove_lora(model)

    assert set(model.state_dict()) == expected_keys
    assert isinstance(model.fc, torch.nn.Linear)


def test_captured_hidden_reuse_is_disabled_by_replay_draft():
    assert _use_captured_hidden("selector", False) is True
    assert _use_captured_hidden("selector", True) is False
    assert _use_captured_hidden("distribution", False) is False
    assert _use_captured_hidden("hybrid", False) is False



