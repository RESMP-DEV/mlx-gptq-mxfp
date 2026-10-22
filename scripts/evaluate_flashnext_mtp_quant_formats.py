#!/usr/bin/env python3
"""Compare native GGML low-bit repacks of the Flash-Next MTP QAT candidate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from mlx_gptq.flashnext_mtp import MixedNativeMTPPairDataset, NativeMTPBank
from mlx_gptq.flashnext_mtp_eval import evaluate, load_candidate, paired_test, per_pair
from mlx_gptq.flashnext_mtp_model import FlashNextMTP
from mlx_gptq.ggml_native import GGML_FORMATS, NativeGGMLQuantizer
from safetensors.torch import load_file

DEFAULT_FORMATS = ("iq4_xs", "iq4_nl", "q4_0", "q4_k", "iq3_s")


def module_names(qat_path: Path) -> list[str]:
    tensors = load_file(str(qat_path))
    return sorted(key[: -len(".q")] for key in tensors if key.endswith(".q"))


def selected_format(name: str, columns: int, requested: str) -> str:
    target = GGML_FORMATS[requested]
    if columns % target.block_size == 0:
        return requested
    if target.block_size != 256 or columns % GGML_FORMATS["iq4_nl"].block_size:
        raise ValueError(f"{requested} cannot cover columns {columns} for {name}")
    return "iq4_nl"


def run(arguments: argparse.Namespace) -> int:
    started = time.time()
    dtype = getattr(torch, arguments.dtype)
    banks = [NativeMTPBank(path) for path in arguments.bank]
    dataset = MixedNativeMTPPairDataset(banks)
    indices = list(range(len(dataset)))
    model = FlashNextMTP.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        dtype=dtype,
        attention_implementation=arguments.attention,
    )
    names = module_names(arguments.qat)
    modules = dict(model.named_modules())
    for name in names:
        if not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"{name} is not linear")

    base_metrics = evaluate(model, dataset, indices, arguments.device, dtype)
    base_records = per_pair(model, dataset, indices, arguments.device, dtype)
    load_candidate(model, arguments.lora, arguments.qat)
    effective = {
        name: modules[name].weight.detach().cpu().to(torch.float32).clone()
        for name in names
    }
    mxfp4_metrics = evaluate(model, dataset, indices, arguments.device, dtype)
    mxfp4_records = per_pair(model, dataset, indices, arguments.device, dtype)

    quantizer = NativeGGMLQuantizer(str(arguments.library))
    total_weights = sum(weight.numel() for weight in effective.values())
    results: list[dict[str, Any]] = []
    for requested in arguments.format:
        format_started = time.time()
        module_results: list[dict[str, Any]] = []
        weighted_bits = 0.0
        for name in names:
            source = effective[name]
            chosen = selected_format(name, int(source.shape[-1]), requested)
            dequantized, statistics = quantizer.quantize_dequantize(
                source, GGML_FORMATS[chosen]
            )
            weighted_bits += source.numel() * GGML_FORMATS[chosen].bits_per_weight
            module_results.append(statistics)
            with torch.no_grad():
                modules[name].weight.copy_(
                    dequantized.to(modules[name].weight.dtype).to(
                        modules[name].weight.device
                    )
                )
        metrics = evaluate(model, dataset, indices, arguments.device, dtype)
        records = per_pair(model, dataset, indices, arguments.device, dtype)
        fallbacks = [
            module["format"]
            for module in module_results
            if module["format"] != requested
        ]
        fallback_counts = {
            name: fallbacks.count(name) for name in sorted(set(fallbacks))
        }
        effective_format = requested
        if fallback_counts:
            effective_format += "+" + ",".join(
                f"{name}x{count}" for name, count in fallback_counts.items()
            )
        results.append(
            {
                "requested_format": requested,
                "effective_format": effective_format,
                "effective_bits_per_weight": weighted_bits / total_weights,
                "total_weights": total_weights,
                "metrics": metrics,
                "versus_base": paired_test(base_records, records),
                "versus_mxfp4": paired_test(mxfp4_records, records),
                "modules": module_results,
                "elapsed_seconds": time.time() - format_started,
            }
        )
        print(json.dumps(results[-1], indent=2))

    receipt = {
        "schema_version": 1,
        "operator": "Codex",
        "device": arguments.device,
        "dtype": arguments.dtype,
        "attention": arguments.attention,
        "pairs": len(dataset),
        "banks": [bank.summary(hashes=True) for bank in banks],
        "library": str(arguments.library),
        "formats": {name: format_.__dict__ for name, format_ in GGML_FORMATS.items()},
        "base": base_metrics,
        "mxfp4": mxfp4_metrics,
        "results": results,
        "elapsed_seconds": time.time() - started,
        "non_claims": [
            "This compares teacher-logit agreement, not live speculative acceptance.",
            "The 256-wide formats use IQ4_NL for unaligned matrix columns.",
            "Native reference quantizers use no importance matrix.",
            "No complete Strata runtime artifact or live serving rollout is produced.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--qat", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path, action="append")
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--format", action="append", choices=sorted(GGML_FORMATS))
    arguments = parser.parse_args()
    if not arguments.format:
        arguments.format = list(DEFAULT_FORMATS)
    return arguments


def main() -> int:
    return run(build_parser())


if __name__ == "__main__":
    raise SystemExit(main())
