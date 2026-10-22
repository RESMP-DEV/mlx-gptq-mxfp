"""MXFP4 QAT tuning for the native Flash-Next MTP target."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .dflash2_distill import _mxfp4_parameters
from .flashnext_mtp import (
    MixedNativeMTPPairDataset,
    NativeMTPBank,
    NativeMTPPairDataset,
)
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


def clamped_teacher_probability(
    teacher: torch.Tensor, confidence_floor: float
) -> torch.Tensor:
    """Return the teacher distribution with a confidence floor and a clamp.

    A verifier row is one sample of a very peaked distribution.  Training
    against that peak teaches the head to reproduce one argmax trajectory
    rather than the teacher's shape, which is what makes a single-prompt fit
    look far better on its own prompt than on anything else.  A confidence
    floor keeps at least ``confidence_floor`` of the mass on the top tokens,
    so the objective is the teacher's shape rather than one point mass, and
    the clamp bounds the per-token gradient contribution.
    """
    logits = teacher.float()
    top = logits.max(dim=-1, keepdim=True).values
    # Cap how far any other logit may sit below the top one.  This bounds the
    # effective distribution support, so one extreme row cannot dominate the
    # gradient.
    clamped = logits.clamp(min=top - _LOGIT_CLAMP_RANGE)
    probabilities = torch.softmax(clamped, dim=-1)
    if confidence_floor <= 0.0:
        return probabilities
    # Flatten the peak by mixing toward the uniform distribution over the
    # tokens that carry any mass.  A verifier row is one sample of a very
    # peaked distribution, and training against that peak teaches the head
    # one argmax trajectory instead of the teacher's shape.  Mixing rather
    # than redistributing keeps the result a proper distribution.
    support = (probabilities > 0).to(probabilities.dtype)
    support_size = support.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform = support / support_size
    top_probability = probabilities.max(dim=-1, keepdim=True).values
    # Solve for the uniform mix that puts the top mass exactly at the floor.
    # With a uniform mass of 1/size, the mixed top mass is
    #   (1 - m) * top + m / size = floor,
    # so m = (top - floor) / (top - 1 / size).
    uniform_top = 1.0 / support_size
    denominator = (top_probability - uniform_top).clamp(min=1e-12)
    mix = ((top_probability - confidence_floor) / denominator).clamp(0.0, 1.0)
    mixed = (1.0 - mix) * probabilities + mix * uniform
    return mixed / mixed.sum(dim=-1, keepdim=True)


_LOGIT_CLAMP_RANGE = 30.0


def loss_and_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ce_weight: float,
    tv_weight: float,
    confidence_floor: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    student_log_prob = torch.log_softmax(student.float(), dim=-1)
    teacher_prob = clamped_teacher_probability(teacher, confidence_floor)
    student_prob = torch.exp(student_log_prob)
    ce = -(teacher_prob * student_log_prob).sum(dim=-1)
    tv = 0.5 * (teacher_prob - student_prob).abs().sum(dim=-1)
    loss = ce_weight * ce + tv_weight * tv
    metrics = distribution_metrics(student, teacher)
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def anchor_regularizer(
    student: torch.Tensor, base: torch.Tensor, weight: float
) -> torch.Tensor:
    """Penalize drift from the frozen base distribution.

    The base head is already a good drafter for the target.  Letting the
    adapter move far from it is what buys a large same-prompt gain and a
    small held-out gain, so an explicit anchor on the base distribution
    trades same-prompt fit for held-out fit in a controlled way.
    """
    if weight <= 0.0:
        return student.new_zeros(())
    student_log_prob = torch.log_softmax(student.float(), dim=-1)
    base_prob = torch.softmax(base.float(), dim=-1)
    return (base_prob * (torch.log(base_prob + 1e-30) - student_log_prob)).sum(dim=-1)


def cross_prompt_contrast(
    student: torch.Tensor,
    teacher: torch.Tensor,
    negative_teacher: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Push the head toward its own prompt's teacher, away from another's.

    A head that has memorized one prompt can fit that prompt's teacher almost
    perfectly while being far from every other prompt's teacher.  Comparing the
    head's own log-probability under the two teachers, on the same row, makes
    that failure visible to the optimizer: the head has to be closer to the
    teacher of the prompt it is being trained on than to the teacher of a
    different prompt.  A head that genuinely learned the target's draft
    distribution satisfies this for free.
    """
    if margin <= 0.0:
        return student.new_zeros(())
    student_log_prob = torch.log_softmax(student.float(), dim=-1)
    own = (teacher.float() * student_log_prob).sum(dim=-1)
    other = (negative_teacher.float() * student_log_prob).sum(dim=-1)
    return torch.clamp(margin - (own - other), min=0.0).mean()


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


def split_banks(
    bank_count: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Hold out whole banks, so validation measures unseen prompts.

    Splitting rows at random would let a bank contribute to both sides, and a
    prompt's rows are strongly correlated, so a row-level split reports the
    training prompt's own fit rather than generalization.  Holding out whole
    banks makes the validation set a set of prompts the head never saw.
    """
    if bank_count < 2:
        raise ValueError("a per-bank validation split needs at least two banks")
    indices = list(range(bank_count))
    random.Random(seed).shuffle(indices)
    validation_count = max(
        1, min(bank_count - 1, int(bank_count * validation_fraction))
    )
    validation = sorted(indices[:validation_count])
    training = sorted(indices[validation_count:])
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
    parser.add_argument(
        "--bank",
        required=True,
        type=Path,
        action="append",
        help=(
            "A native MTP teacher bank. Repeat the flag to mix several banks "
            "into one training set."
        ),
    )
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
        "--gradient-checkpointing",
        action="store_true",
        help="Rematerialize the native MTP forward during backward.",
    )
    parser.add_argument(
        "--fp16-initial-scale",
        type=float,
        default=65536.0,
        help="Initial dynamic-loss scale for float16 CUDA training.",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum teacher probability mass on the top tokens. 0 keeps the "
            "exact verifier distribution."
        ),
    )
    parser.add_argument(
        "--contrast-margin",
        type=float,
        default=0.0,
        help=(
            "Margin for the cross-prompt contrast. 0 disables the term. The "
            "head must be closer to its own prompt's teacher than to another "
            "prompt's teacher by this margin."
        ),
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help=("Weight on the KL anchor to the frozen base head. 0 disables it."),
    )
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--hold-out-banks",
        action="store_true",
        help=(
            "Hold out whole banks for validation so the validation set is a "
            "set of prompts the head never saw."
        ),
    )
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

    banks = [NativeMTPBank(Path(path)) for path in arguments.bank]
    bank = banks[0] if len(banks) == 1 else None
    dataset: Any = (
        NativeMTPPairDataset(banks[0])
        if len(banks) == 1
        else MixedNativeMTPPairDataset(banks)
    )
    selected = (
        list(range(len(dataset)))
        if arguments.limit <= 0
        else list(range(min(arguments.limit, len(dataset))))
    )
    if arguments.hold_out_banks and arguments.limit > 0:
        raise ValueError(
            "--hold-out-banks cannot be combined with --limit: limiting rows "
            "cuts across bank boundaries, so the held-out banks would no "
            "longer be whole prompts"
        )
    if arguments.hold_out_banks:
        # Hold out whole banks so validation measures unseen prompts.
        pairs_per_bank = (
            dataset.per_bank_counts()
            if isinstance(dataset, MixedNativeMTPPairDataset)
            else [len(dataset)]
        )
        training_banks, validation_banks = split_banks(
            len(pairs_per_bank), arguments.validation_fraction, arguments.seed
        )
        train_indices = []
        validation_indices = []
        for bank_index in training_banks:
            start = sum(pairs_per_bank[:bank_index])
            train_indices.extend(range(start, start + pairs_per_bank[bank_index]))
        for bank_index in validation_banks:
            start = sum(pairs_per_bank[:bank_index])
            validation_indices.extend(range(start, start + pairs_per_bank[bank_index]))
        train_indices = [
            selected[index] for index in train_indices if index < len(selected)
        ]
        validation_indices = [
            selected[index] for index in validation_indices if index < len(selected)
        ]
        if not train_indices or not validation_indices:
            raise ValueError(
                "the per-bank hold-out left no training or validation pairs"
            )
    else:
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
    base_head = None
    if arguments.anchor_weight > 0.0:
        # The base head is the frozen reference the anchor pulls toward.  It is
        # captured before the adapter is folded in, so it is the true base.
        base_head = copy.deepcopy(model).to(arguments.device)
        base_head.eval()
        base_head.requires_grad_(False)
    model.gradient_checkpointing = arguments.gradient_checkpointing
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
                    student,
                    teacher,
                    arguments.ce_weight,
                    arguments.tv_weight,
                    arguments.confidence_floor,
                )
                if arguments.contrast_margin > 0.0 and isinstance(
                    dataset, MixedNativeMTPPairDataset
                ):
                    negative = dataset.negative_index(index, arguments.seed)
                    negative_teacher = dataset[negative]["target_logits"]
                    negative_teacher = negative_teacher.unsqueeze(0).to(
                        arguments.device
                    )
                    loss = loss + cross_prompt_contrast(
                        student,
                        teacher,
                        negative_teacher,
                        arguments.contrast_margin,
                    )
                if base_head is not None:
                    with torch.no_grad():
                        base_logits = base_head(residual, tokens, positions=positions)[
                            0
                        ]
                    loss = loss + anchor_regularizer(
                        student, base_logits, arguments.anchor_weight
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
        "bank": (
            bank.summary(hashes=True)
            if bank is not None
            else {
                "mix": True,
                "banks": [item.summary(hashes=True) for item in banks],
                "pairs_per_bank": (
                    dataset.per_bank_counts()
                    if isinstance(dataset, MixedNativeMTPPairDataset)
                    else [len(dataset)]
                ),
            }
        ),
        "training_pairs": len(train_indices),
        "validation_pairs": len(validation_indices),
        "steps": step,
        "stopped_by_step_cap": stopped_by_step_cap,
        "qat_mxfp4": arguments.qat_mxfp4,
        "confidence_floor": arguments.confidence_floor,
        "anchor_weight": arguments.anchor_weight,
        "contrast_margin": arguments.contrast_margin,
        "bank_count": len(banks),
        "hold_out_banks": arguments.hold_out_banks,
        "gradient_checkpointing": arguments.gradient_checkpointing,
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
