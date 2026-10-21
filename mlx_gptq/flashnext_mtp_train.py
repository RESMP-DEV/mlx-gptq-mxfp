"""MXFP4 QAT tuning for the native Flash-Next MTP target."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .dflash2_distill import _mxfp4_parameters
from .flashnext_mtp import NativeMTPBank, NativeMTPPairDataset
from .flashnext_mtp_model import FlashNextMTP
from .qat import fake_quantize_mxfp4


class FlashNextLoRALinear(torch.nn.Module):
    """Float32 LoRA master over a frozen MTP linear weight."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, qat_mxfp4: bool):
        super().__init__()
        if rank <= 0:
            raise ValueError("Flash-Next MTP LoRA rank must be positive")
        if base.bias is not None:
            raise ValueError("Flash-Next MTP LoRA requires bias-free linear layers")
        if base.weight.shape[-1] % 32:
            raise ValueError("Flash-Next MTP QAT input width must be divisible by 32")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.qat_mxfp4 = bool(qat_mxfp4)
        self.a = torch.nn.Parameter(
            torch.empty(
                rank, base.in_features, dtype=torch.float32, device=base.weight.device
            )
        )
        self.b = torch.nn.Parameter(
            torch.empty(
                base.out_features,
                rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )
        torch.nn.init.kaiming_uniform_(self.a, a=5**0.5)
        torch.nn.init.zeros_(self.b)

    def effective_weight(self) -> torch.Tensor:
        effective = self.base.weight.float() + (self.b @ self.a) * self.scaling
        return fake_quantize_mxfp4(effective) if self.qat_mxfp4 else effective

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            input, self.effective_weight().to(input.dtype)
        )


TUNED_LINEAR_PREFIXES = (
    "fc_embedding",
    "fc_hidden",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
    "attn_hyper_connection.input_mix_weight_down",
    "attn_hyper_connection.input_mix_weight_up",
    "attn_hyper_connection.block_inject_weight",
    "mlp_hyper_connection.input_mix_weight_down",
    "mlp_hyper_connection.input_mix_weight_up",
    "mlp_hyper_connection.block_inject_weight",
    "hyper_connection_mixer.input_mix_weight_down",
    "hyper_connection_mixer.input_mix_weight_up",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)


def install_lora(
    model: torch.nn.Module, rank: int, alpha: float, qat_mxfp4: bool
) -> tuple[list[str], int]:
    """Replace only the selected dense MTP projections and freeze the base."""
    replaced: list[str] = []

    def visit(module: torch.nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch.nn.Linear) and name.startswith(
                TUNED_LINEAR_PREFIXES
            ):
                adapter = FlashNextLoRALinear(child, rank, alpha, qat_mxfp4)
                setattr(module, child_name, adapter)
                replaced.append(name)
                continue
            visit(child, name)

    visit(model)
    expected = len(TUNED_LINEAR_PREFIXES)
    if len(replaced) != expected:
        raise RuntimeError(
            f"installed {len(replaced)} Flash-Next MTP LoRA modules, "
            f"expected {expected}"
        )
    return sorted(replaced)


def distribution_metrics(
    student: torch.Tensor, teacher: torch.Tensor
) -> dict[str, float | int]:
    """Return the full-vocabulary teacher-agreement metrics."""
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    student = student.detach().float()
    teacher = teacher.detach().float()
    student_log_prob = torch.log_softmax(student, dim=-1)
    teacher_prob = torch.softmax(teacher, dim=-1)
    student_prob = torch.exp(student_log_prob)
    ce = -(teacher_prob * student_log_prob).sum(dim=-1)
    tv = 0.5 * (teacher_prob - student_prob).abs().sum(dim=-1)
    kl = (teacher_prob * (torch.log(teacher_prob + 1e-30) - student_log_prob)).sum(
        dim=-1
    )
    return {
        "loss_ce": float(ce.detach()),
        "loss_tv": float(tv.detach()),
        "loss_kl": float(kl.detach()),
        "student_top": float(student.max()),
        "teacher_top": float(teacher.max()),
        "student_argmax": int(student.argmax()),
        "teacher_argmax": int(teacher.argmax()),
        "top1_match": int(student.argmax()) == int(teacher.argmax()),
    }


def loss_and_metrics(
    student: torch.Tensor, teacher: torch.Tensor, ce_weight: float, tv_weight: float
) -> tuple[torch.Tensor, dict[str, float | int]]:
    student_log_prob = torch.log_softmax(student.float(), dim=-1)
    teacher_prob = torch.softmax(teacher.float(), dim=-1)
    student_prob = torch.exp(student_log_prob)
    ce = -(teacher_prob * student_log_prob).sum(dim=-1)
    tv = 0.5 * (teacher_prob - student_prob).abs().sum(dim=-1)
    loss = ce_weight * ce + tv_weight * tv
    metrics = distribution_metrics(student, teacher)
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def start_wandb(arguments: argparse.Namespace, output: Path) -> Any | None:
    if arguments.wandb_mode == "disabled":
        return None
    os.environ.setdefault("WANDB_DIR", str(output))
    if arguments.wandb_mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    return wandb.init(
        project=arguments.wandb_project,
        name=arguments.wandb_name,
        mode=arguments.wandb_mode,
        dir=str(output),
        tags=[item for item in arguments.wandb_tags.split(",") if item],
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
    )


def export_artifacts(
    model: torch.nn.Module, output: Path, replaced: list[str]
) -> dict[str, int | bool]:
    """Export float LoRA plus the exact effective MXFP4 grid."""
    adapters: dict[str, torch.Tensor] = {}
    qat: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, FlashNextLoRALinear):
            continue
        adapters[f"{name}.a"] = module.a.detach().cpu().contiguous()
        adapters[f"{name}.b"] = module.b.detach().cpu().contiguous()
        adapters[f"{name}.scaling"] = torch.tensor(module.scaling, dtype=torch.float32)
        effective = module.effective_weight()
        if module.qat_mxfp4:
            codes, scales, _effective = _mxfp4_parameters(effective)
            qat[f"{name}.q"] = codes.cpu().contiguous()
            qat[f"{name}.scales"] = scales.cpu().contiguous()
    save_file(adapters, str(output / "flashnext_mtp_lora.safetensors"))
    if qat:
        save_file(qat, str(output / "flashnext_mtp_mxfp4_qat.safetensors"))
    return {
        "lora_tensors": len(adapters),
        "qat_tensors": len(qat),
        "qat_artifact": bool(qat),
        "tuned_modules": len(replaced),
    }


def split_pairs(
    count: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, min(count - 1, int(count * validation_fraction)))
    validation = sorted(indices[:validation_count])
    training = sorted(indices[validation_count:])
    if not training:
        raise ValueError("Flash-Next MTP split left no training pairs")
    return training, validation


def mtp_batch(
    pair: dict[str, torch.Tensor], device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one independent MTP pair while preserving its absolute position."""
    return (
        pair["residual"].view(1, 1, -1).to(device, dtype=dtype),
        pair["input_token"].view(1, 1).to(device),
        pair["source_position"].view(1, 1).to(device),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=0, help="Hard optimizer-step cap; 0 is uncapped."
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--qat-mxfp4", action="store_true")
    parser.add_argument(
        "--fp16-initial-scale",
        type=float,
        default=65536.0,
        help="Initial dynamic-loss scale for float16 CUDA training.",
    )
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="offline"
    )
    parser.add_argument("--wandb-project", default="flashnext-mtp-mxfp4-qat")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", default="flashnext,mtp,mxfp4,qat")
    return parser


def run(arguments: argparse.Namespace) -> int:
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, arguments.dtype)

    bank = NativeMTPBank(Path(arguments.bank))
    dataset = NativeMTPPairDataset(bank)
    selected = (
        list(range(len(dataset)))
        if arguments.limit <= 0
        else list(range(min(arguments.limit, len(dataset))))
    )
    train_indices, validation_indices = split_pairs(
        len(selected), arguments.validation_fraction, arguments.seed
    )
    train_indices = [selected[index] for index in train_indices]
    validation_indices = [selected[index] for index in validation_indices]

    model = FlashNextMTP.from_checkpoint(
        Path(arguments.checkpoint),
        device=arguments.device,
        dtype=dtype,
        attention_implementation=arguments.attention,
    )
    replaced = install_lora(
        model, arguments.lora_rank, arguments.lora_alpha, arguments.qat_mxfp4
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        arguments.device.split(":")[0],
        enabled=arguments.dtype == "float16" and arguments.device.startswith("cuda"),
        init_scale=arguments.fp16_initial_scale,
    )
    wandb_run = start_wandb(arguments, output)
    started = time.time()
    history: list[dict[str, Any]] = []
    step = 0
    stopped_by_step_cap = False
    overflow_count = 0

    def evaluate(indices: list[int]) -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        with torch.no_grad():
            for index in indices:
                pair = dataset[index]
                residual, tokens, positions = mtp_batch(pair, arguments.device, dtype)
                teacher = pair["target_logits"].unsqueeze(0).to(arguments.device)
                student = model(residual, tokens, positions=positions)[0]
                metrics = distribution_metrics(student, teacher)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
        return {key: value / len(indices) for key, value in totals.items()}

    initial_validation = evaluate(validation_indices)
    model.train()
    try:
        for _epoch in range(arguments.epochs):
            order = list(train_indices)
            random.Random(arguments.seed + step).shuffle(order)
            for index in order:
                pair = dataset[index]
                residual, tokens, positions = mtp_batch(pair, arguments.device, dtype)
                teacher = pair["target_logits"].unsqueeze(0).to(arguments.device)
                optimizer.zero_grad(set_to_none=True)
                student = model(residual, tokens, positions=positions)[0]
                loss, metrics = loss_and_metrics(
                    student, teacher, arguments.ce_weight, arguments.tv_weight
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    arguments.max_grad_norm,
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    overflow_count += 1
                step += 1
                record = {
                    "step": step,
                    "pair_index": index,
                    "source_position": int(pair["source_position"]),
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm.detach()),
                    "loss_scale": float(scaler.get_scale()),
                    "overflow_count": overflow_count,
                    **metrics,
                }
                history.append(record)
                if wandb_run is not None:
                    wandb_run.log(record)
                if arguments.steps and step >= arguments.steps:
                    stopped_by_step_cap = True
                    break
            if stopped_by_step_cap:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    final_validation = evaluate(validation_indices)
    export = export_artifacts(model, output, replaced)
    elapsed = time.time() - started
    receipt = {
        "schema_version": 1,
        "operator": "Codex",
        "elapsed_seconds": elapsed,
        "device": arguments.device,
        "dtype": arguments.dtype,
        "attention": arguments.attention,
        "bank": bank.summary(hashes=True),
        "training_pairs": len(train_indices),
        "validation_pairs": len(validation_indices),
        "steps": step,
        "stopped_by_step_cap": stopped_by_step_cap,
        "qat_mxfp4": arguments.qat_mxfp4,
        "fp16_initial_scale": arguments.fp16_initial_scale,
        "final_loss_scale": float(scaler.get_scale()),
        "overflow_count": overflow_count,
        "tuned_modules": len(replaced),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "export": export,
        "wandb": (
            None
            if wandb_run is None
            else {
                "mode": arguments.wandb_mode,
                "project": arguments.wandb_project,
                "name": wandb_run.name,
                "id": wandb_run.id,
                "dir": str(wandb_run.dir),
            }
        ),
        "history": history,
        "non_claims": [
            "Validation pairs come from one private prompt and are not "
            "session-disjoint acceptance.",
            "Offline W&B logging does not prove online synchronization.",
            "MXFP4 QAT export proves the effective grid, not runtime "
            "integration or speed.",
            "No candidate is promoted to Strata serving.",
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
