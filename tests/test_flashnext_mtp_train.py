from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mlx_gptq.flashnext_mtp_train import (
    FlashNextLoRALinear,
    distribution_metrics,
    export_artifacts,
    install_lora,
    loss_and_metrics,
    mtp_batch,
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
