import pytest
import torch

from mlx_gptq.dflash2_distill import (
    LoRALinear,
    _forward,
    _install_lora,
    _prompt_split,
    _selector_loss,
)


class _Selector:
    def __init__(self):
        self.top_k = 2
        self.predecessor_codebook = torch.nn.Embedding(10, 4)
        self.successor_codebook = torch.nn.Embedding(10, 4)
        self.hidden_projection = torch.nn.Linear(4, 4, bias=False)


class _Model:
    def __init__(self):
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


def test_forward_falls_back_when_raw_context_is_incomplete():
    class Rotary:
        def __call__(self, noise, position_ids):
            return torch.zeros_like(noise), torch.zeros_like(noise)

    class Layer(torch.nn.Module):
        def forward(self, hidden_states, target_hidden, **_):
            assert target_hidden.shape[1] == 3
            return hidden_states

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4, bias=False)
            self.hidden_norm = torch.nn.LayerNorm(4)
            self.rotary_emb = Rotary()
            self.layers = [Layer()]
            self.norm = torch.nn.Identity()

    payload = {
        "context": torch.randn(1, 3, 4),
        "raw_context": torch.randn(1, 1, 8),
        "noise_embedding": torch.randn(1, 2, 4),
    }
    output = _forward(Model(), payload, "cpu")

    assert output.shape == (1, 1, 4)


def test_prompt_split_never_leaks_a_prompt():
    prompt_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    train, validation, validation_prompts = _prompt_split(
        prompt_ids, fraction=0.25, seed=17
    )

    train_prompts = {prompt_ids[index] for index in train}
    assert train_prompts.isdisjoint(validation_prompts)
    assert set(validation_prompts) == {prompt_ids[index] for index in validation}
    assert len(train) + len(validation) == len(prompt_ids)
