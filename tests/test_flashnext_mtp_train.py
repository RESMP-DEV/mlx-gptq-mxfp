from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mlx_gptq.flashnext_mtp_model import FlashNextMTP
from mlx_gptq.flashnext_mtp_train import (
    FlashNextLoRALinear,
    anchor_regularizer,
    clamped_teacher_probability,
    cross_prompt_contrast,
    distribution_metrics,
    export_artifacts,
    install_lora,
    loss_and_metrics,
    mtp_batch,
    split_banks,
    split_pairs,
)


def test_lora_qat_has_float_master_and_straight_through_gradient() -> None:
    base = torch.nn.Linear(64, 32, bias=False).float()
    module = FlashNextLoRALinear(base, rank=4, alpha=8.0, qat_mxfp4=True)
    input = torch.randn(2, 64, requires_grad=True)
    output = module(input)
    output.square().mean().backward()
    assert module.a.dtype == torch.float32
    assert module.b.dtype == torch.float32
    assert module.a.grad is not None
    assert module.b.grad is not None
    assert input.grad is not None
    assert torch.isfinite(output).all()


def test_lora_rank_must_be_positive() -> None:
    base = torch.nn.Linear(64, 32, bias=False).float()
    with pytest.raises(ValueError, match="positive"):
        FlashNextLoRALinear(base, rank=0, alpha=8.0, qat_mxfp4=True)


def test_non_qat_export_does_not_write_qat_artifact(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.layer = FlashNextLoRALinear(
        torch.nn.Linear(64, 32, bias=False), rank=4, alpha=8.0, qat_mxfp4=False
    )
    receipt = export_artifacts(model, tmp_path, ["layer"])
    assert receipt["qat_tensors"] == 0
    assert receipt["qat_artifact"] is False
    assert (tmp_path / "flashnext_mtp_lora.safetensors").is_file()
    assert not (tmp_path / "flashnext_mtp_mxfp4_qat.safetensors").exists()


def test_metrics_and_loss_use_full_teacher_distribution() -> None:
    student = torch.tensor([[1.0, 0.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0, 0.0]])
    metrics = distribution_metrics(student, teacher)
    assert metrics["student_argmax"] == 0
    assert metrics["teacher_argmax"] == 1
    assert metrics["top1_match"] is False
    assert metrics["loss_tv"] > 0.36

    loss, with_loss = loss_and_metrics(student, teacher, 0.1, 0.9)
    assert loss.ndim == 1
    assert loss.shape == (1,)
    assert with_loss["loss"] > with_loss["loss_tv"]


def test_split_is_disjoint_and_covers_selection() -> None:
    train, validation = split_pairs(11, 0.2, seed=73)
    assert len(train) == 9
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(11))


def test_mtp_batch_preserves_absolute_source_position() -> None:
    pair = {
        "residual": torch.arange(8, dtype=torch.float32),
        "input_token": torch.tensor(795, dtype=torch.long),
        "source_position": torch.tensor(511, dtype=torch.long),
    }
    residual, tokens, positions = mtp_batch(pair, "cpu", torch.bfloat16)
    assert residual.shape == (1, 1, 8)
    assert residual.dtype == torch.bfloat16
    assert tokens.equal(torch.tensor([[795]], dtype=torch.long))
    assert positions.equal(torch.tensor([[511]], dtype=torch.long))


def test_gradient_checkpointing_wrapper_propagates_gradients() -> None:
    model = FlashNextMTP.__new__(FlashNextMTP)
    torch.nn.Module.__init__(model)
    model.gradient_checkpointing = True
    model.training = True
    model._forward = lambda residual, _tokens, _positions: residual.square().sum()
    residual = torch.randn(2, requires_grad=True)

    model(residual, torch.zeros(2, dtype=torch.long), torch.zeros(2)).backward()

    assert residual.grad is not None
    assert torch.allclose(residual.grad, 2 * residual)


def test_install_lora_replaces_expected_modules() -> None:
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_embedding = torch.nn.Linear(64, 64, bias=False)
            self.fc_hidden = torch.nn.Linear(64, 64, bias=False)
            self.self_attn = torch.nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.self_attn, name, torch.nn.Linear(64, 64, bias=False))
            self.self_attn.indexer = torch.nn.Module()
            self.self_attn.indexer.index_qk_proj = torch.nn.Linear(64, 64, bias=False)
            self.attn_hyper_connection = torch.nn.Module()
            for name in (
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
            ):
                setattr(
                    self.attn_hyper_connection,
                    name,
                    torch.nn.Linear(64, 64, bias=False),
                )
            self.mlp_hyper_connection = torch.nn.Module()
            for name in (
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
            ):
                setattr(
                    self.mlp_hyper_connection, name, torch.nn.Linear(64, 64, bias=False)
                )
            self.hyper_connection_mixer = torch.nn.Module()
            for name in ("input_mix_weight_down", "input_mix_weight_up"):
                setattr(
                    self.hyper_connection_mixer,
                    name,
                    torch.nn.Linear(64, 64, bias=False),
                )
            self.mlp = torch.nn.Module()
            self.mlp.shared_expert = torch.nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(
                    self.mlp.shared_expert, name, torch.nn.Linear(64, 64, bias=False)
                )

    model = Toy()
    model.requires_grad_(False)
    replaced = install_lora(model, rank=4, alpha=8.0, qat_mxfp4=True)
    assert len(replaced) == 18
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert len(trainable) == 36


def test_confidence_floor_caps_teacher_peakiness() -> None:
    teacher = torch.tensor([[10.0, 0.0, 0.0, -50.0]])
    exact = clamped_teacher_probability(teacher, 0.0)
    assert abs(float(exact.sum()) - 1.0) < 1e-5
    for floor in (0.5, 0.9, 0.99):
        floored = clamped_teacher_probability(teacher, floor)
        assert abs(float(floored.sum()) - 1.0) < 1e-5
        assert float(floored.max()) <= floor + 1e-6
        assert int(floored.argmax()) == int(exact.argmax())


def test_confidence_floor_does_not_change_a_flat_teacher() -> None:
    flat = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    floored = clamped_teacher_probability(flat, 0.5)
    assert torch.allclose(floored, torch.full_like(floored, 0.25), atol=1e-6)


def test_anchor_regularizer_is_zero_for_identical_distributions() -> None:
    student = torch.tensor([[1.0, 0.0]])
    base = torch.tensor([[1.0, 0.0]])
    assert float(anchor_regularizer(student, base, 1.0)) == 0.0
    assert float(anchor_regularizer(student, base, 0.0)) == 0.0


def test_anchor_regularizer_penalizes_drift() -> None:
    student = torch.tensor([[0.0, 1.0]])
    base = torch.tensor([[1.0, 0.0]])
    assert float(anchor_regularizer(student, base, 1.0)) > 0.0


def test_split_banks_holds_out_whole_banks() -> None:
    training, validation = split_banks(9, 0.34, seed=73)
    assert len(training) == 6
    assert len(validation) == 3
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == set(range(9))


def test_split_banks_always_keeps_one_training_bank() -> None:
    training, validation = split_banks(2, 0.5, seed=73)
    assert len(training) == 1
    assert len(validation) == 1
    assert set(training).isdisjoint(validation)


def test_split_banks_needs_at_least_two_banks() -> None:
    try:
        split_banks(1, 0.5, seed=73)
    except ValueError as error:
        assert "at least two banks" in str(error)
    else:
        raise AssertionError("expected a ValueError for a single bank")


def test_cross_prompt_contrast_is_zero_when_the_head_owns_its_teacher() -> None:
    # The student is exactly the own teacher, so the margin is already met.
    teacher = torch.tensor([[2.0, 0.0]])
    negative = torch.tensor([[0.0, 2.0]])
    student = teacher.clone()
    assert float(cross_prompt_contrast(student, teacher, negative, 1.0)) == 0.0


def test_cross_prompt_contrast_penalizes_a_memorized_prompt() -> None:
    # The student matches the negative teacher instead of its own, so the
    # margin is violated and the term is positive.
    teacher = torch.tensor([[2.0, 0.0]])
    negative = torch.tensor([[0.0, 2.0]])
    student = negative.clone()
    assert float(cross_prompt_contrast(student, teacher, negative, 1.0)) > 0.0


def test_cross_prompt_contrast_is_disabled_at_zero_margin() -> None:
    teacher = torch.tensor([[2.0, 0.0]])
    negative = torch.tensor([[0.0, 2.0]])
    student = negative.clone()
    assert float(cross_prompt_contrast(student, teacher, negative, 0.0)) == 0.0
