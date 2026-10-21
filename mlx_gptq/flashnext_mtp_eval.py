"""Evaluate a Flash-Next native MTP candidate against its frozen base.

The candidate is the exact MXFP4 grid the trainer exported, so evaluation
rebuilds the effective weight from the stored E2M1 codes and E8M0 scales
rather than from the float LoRA masters.  That makes the measured numbers a
property of the artifact that would actually be served.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .flashnext_mtp import (
    MixedNativeMTPPairDataset,
    NativeMTPBank,
    NativeMTPPairDataset,
)
from .flashnext_mtp_model import FlashNextMTP
from .flashnext_mtp_train import split_pairs
from .grid import dequantize_mxfp


def load_candidate(
    model: FlashNextMTP, lora_path: Path, qat_path: Path
) -> dict[str, Any]:
    """Fold the exported LoRA into the frozen base and apply the MXFP4 grid."""
    lora = load_file(str(lora_path))
    qat = load_file(str(qat_path))
    applied = 0
    max_absolute = 0.0
    max_relative = 0.0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if f"{name}.a" not in lora:
            continue
        weight_a = lora[f"{name}.a"].float()
        weight_b = lora[f"{name}.b"].float()
        scaling = float(lora[f"{name}.scaling"].float().reshape(-1)[0])
        effective = module.weight.float() + (weight_b @ weight_a) * scaling
        codes = qat[f"{name}.q"]
        scales = qat[f"{name}.scales"]
        dequantized = dequantize_mxfp(codes, scales, "mxfp4").reshape(effective.shape)
        error = (dequantized - effective).abs()
        max_absolute = max(max_absolute, error.max().item())
        max_relative = max(
            max_relative,
            error.max().item() / max(effective.abs().max().item(), 1e-12),
        )
        with torch.no_grad():
            module.weight.copy_(dequantized.to(module.weight.dtype))
        applied += 1
    return {
        "applied_modules": applied,
        "adapter_tensors": len(lora),
        "qat_tensors": len(qat),
        "mxfp4_quantization_max_absolute_error": max_absolute,
        "mxfp4_quantization_max_relative_error": max_relative,
    }


def evaluate(
    model: FlashNextMTP,
    dataset: NativeMTPPairDataset,
    indices: list[int],
    device: str,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Return the mean full-vocabulary teacher-agreement metrics."""
    model.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for index in indices:
            pair = dataset[index]
            residual = pair["residual"].view(1, 1, -1).to(device, dtype=dtype)
            tokens = pair["input_token"].view(1, 1).to(device)
            positions = pair["source_position"].view(1, 1).to(device)
            teacher = pair["target_logits"].unsqueeze(0).to(device)
            student = model(residual, tokens, positions=positions)[0]
            student_log_prob = torch.log_softmax(student.float(), dim=-1)
            teacher_prob = torch.softmax(teacher.float(), dim=-1)
            student_prob = torch.exp(student_log_prob)
            ce = -(teacher_prob * student_log_prob).sum(dim=-1)
            tv = 0.5 * (teacher_prob - student_prob).abs().sum(dim=-1)
            kl = (
                teacher_prob * (torch.log(teacher_prob + 1e-30) - student_log_prob)
            ).sum(dim=-1)
            values = {
                "loss_ce": float(ce),
                "loss_tv": float(tv),
                "loss_kl": float(kl),
                "top1_match": float(int(student.argmax()) == int(teacher.argmax())),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(indices) for key, value in totals.items()}


def per_pair(
    model: FlashNextMTP,
    dataset: NativeMTPPairDataset,
    indices: list[int],
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, float]]:
    """Return one record per evaluated pair for paired statistics."""
    model.eval()
    records: list[dict[str, float]] = []
    with torch.no_grad():
        for index in indices:
            pair = dataset[index]
            residual = pair["residual"].view(1, 1, -1).to(device, dtype=dtype)
            tokens = pair["input_token"].view(1, 1).to(device)
            positions = pair["source_position"].view(1, 1).to(device)
            teacher = pair["target_logits"].unsqueeze(0).to(device)
            student = model(residual, tokens, positions=positions)[0]
            student_log_prob = torch.log_softmax(student.float(), dim=-1)
            teacher_prob = torch.softmax(teacher.float(), dim=-1)
            student_prob = torch.exp(student_log_prob)
            records.append(
                {
                    "bank_index": float(pair.get("bank_index", torch.tensor(0)).item()),
                    "index": float(index),
                    "source_position": float(int(pair["source_position"])),
                    "loss_ce": float(-(teacher_prob * student_log_prob).sum(dim=-1)),
                    "loss_tv": float(
                        0.5 * (teacher_prob - student_prob).abs().sum(dim=-1)
                    ),
                    "top1_match": float(int(student.argmax()) == int(teacher.argmax())),
                }
            )
    return records


def paired_test(
    base: list[dict[str, float]], candidate: list[dict[str, float]]
) -> dict[str, Any]:
    """Exact McNemar for top-1 and a paired bootstrap for the CE difference."""
    if len(base) != len(candidate):
        raise ValueError("paired records must be the same length")
    base_top = [record["top1_match"] > 0.5 for record in base]
    candidate_top = [record["top1_match"] > 0.5 for record in candidate]
    wins = sum(1 for b, c in zip(base_top, candidate_top) if c and not b)
    losses = sum(1 for b, c in zip(base_top, candidate_top) if b and not c)
    ties = len(base_top) - wins - losses
    total = wins + losses
    if total == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test on the discordant pairs.
        k = min(wins, losses)
        cumulative = sum(_binomial_coefficient(total, i) for i in range(0, k + 1))
        p_value = min(1.0, 2.0 * cumulative / (2**total))
    differences = [
        candidate_record["loss_ce"] - base_record["loss_ce"]
        for base_record, candidate_record in zip(base, candidate)
    ]
    mean_difference = sum(differences) / len(differences)
    generator = torch.Generator().manual_seed(20261021)
    count = len(differences)
    samples = torch.randint(0, count, (2000, count), generator=generator)
    stacked = torch.tensor(differences)
    means = stacked[samples].mean(dim=1)
    lower = float(torch.quantile(means, 0.025))
    upper = float(torch.quantile(means, 0.975))
    return {
        "pairs": count,
        "top1_candidate_wins": wins,
        "top1_candidate_losses": losses,
        "top1_ties": ties,
        "mcnemar_exact_p_value": p_value,
        "ce_mean_difference": mean_difference,
        "ce_difference_ci95": [lower, upper],
        "ce_difference_ci95_excludes_zero": lower > 0.0 or upper < 0.0,
    }


def _binomial_coefficient(n: int, k: int) -> int:
    result = 1
    for index in range(k):
        result = result * (n - index) // (index + 1)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--bank",
        required=True,
        type=Path,
        action="append",
        help="A native MTP bank; repeat to evaluate a mixed multi-prompt set.",
    )
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--qat", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="eager")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Mirror the trainer's split to cross-check its receipt.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=73)
    return parser


def run(arguments: argparse.Namespace) -> int:
    started = time.time()
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
    if arguments.split == "all":
        indices = selected
    else:
        train_indices, validation_indices = split_pairs(
            len(selected), arguments.validation_fraction, arguments.seed
        )
        chosen = train_indices if arguments.split == "train" else validation_indices
        indices = [selected[index] for index in chosen]
    model = FlashNextMTP.from_checkpoint(
        Path(arguments.checkpoint),
        device=arguments.device,
        dtype=dtype,
        attention_implementation=arguments.attention,
    )
    base = evaluate(model, dataset, indices, arguments.device, dtype)
    has_candidate = bool(arguments.lora and arguments.qat)
    base_records = (
        per_pair(model, dataset, indices, arguments.device, dtype)
        if has_candidate
        else []
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "operator": "Codex",
        "device": arguments.device,
        "dtype": arguments.dtype,
        "attention": arguments.attention,
        "pairs": len(indices),
        "split": arguments.split,
        "validation_fraction": arguments.validation_fraction,
        "seed": arguments.seed,
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
        "base": base,
    }
    if has_candidate:
        result["candidate_load"] = load_candidate(model, arguments.lora, arguments.qat)
        result["candidate"] = evaluate(model, dataset, indices, arguments.device, dtype)
        result["delta"] = {key: result["candidate"][key] - base[key] for key in base}
        candidate_records = per_pair(model, dataset, indices, arguments.device, dtype)
        result["paired_test"] = paired_test(base_records, candidate_records)
    result["elapsed_seconds"] = time.time() - started
    result["non_claims"] = [
        "This measures teacher-logit agreement, not live speculative acceptance.",
        "The bank is one private prompt and is not a multi-prompt corpus.",
    ]
    Path(arguments.output).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
