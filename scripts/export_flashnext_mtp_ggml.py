#!/usr/bin/env python3
"""Export the effective Flash-Next MTP candidate as native GGML blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from mlx_gptq.flashnext_mtp_eval import load_candidate
from mlx_gptq.flashnext_mtp_model import FlashNextMTP
from mlx_gptq.ggml_native import GGML_FORMATS, NativeGGMLQuantizer
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_candidate(
    checkpoint: Path,
    lora: Path,
    qat: Path,
    library: Path,
    output: Path,
    requested_format: str,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"GGML output already exists: {output}")
    model = FlashNextMTP.from_checkpoint(
        checkpoint, device="cpu", dtype=torch.float32, attention_implementation="sdpa"
    )
    load_candidate(model, lora, qat)
    qat_tensors = load_file(str(qat))
    modules = dict(model.named_modules())
    quantizer = NativeGGMLQuantizer(str(library))
    packed: dict[str, torch.Tensor] = {}
    module_formats: dict[str, str] = {}
    module_statistics: list[dict[str, object]] = []
    weighted_bits = 0.0
    total_weights = 0
    for key in sorted(qat_tensors):
        if not key.endswith(".q"):
            continue
        name = key[: -len(".q")]
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"QAT tensor {key} has no matching linear module")
        weight = module.weight.detach().cpu()
        target = GGML_FORMATS[requested_format]
        columns = int(weight.shape[-1])
        if columns % target.block_size == 0:
            chosen = requested_format
        elif (
            target.block_size == 256
            and columns % GGML_FORMATS["iq4_nl"].block_size == 0
        ):
            chosen = "iq4_nl"
        else:
            raise ValueError(
                f"{requested_format} cannot cover columns {columns} for {name}"
            )
        blocks, statistics = quantizer.quantize_blocks(weight, GGML_FORMATS[chosen])
        packed[f"{name}.blocks"] = blocks.contiguous()
        module_formats[name] = chosen
        module_statistics.append(statistics)
        weighted_bits += weight.numel() * GGML_FORMATS[chosen].bits_per_weight
        total_weights += weight.numel()
    if not packed:
        raise ValueError("the QAT artifact contains no module grids")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        packed,
        str(output),
        metadata={
            "requested_format": requested_format,
            "module_formats": json.dumps(module_formats, sort_keys=True),
        },
    )
    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "lora_sha256": sha256(lora),
        "qat_sha256": sha256(qat),
        "library": str(library),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "requested_format": requested_format,
        "module_formats": module_formats,
        "effective_bits_per_weight": weighted_bits / total_weights,
        "total_weights": total_weights,
        "modules": module_statistics,
        "non_claims": [
            "This is a packed weight artifact, not a complete Strata rt_dir.",
            "Teacher-logit evaluation and live acceptance are still required.",
            "The 256-wide formats use IQ4_NL for unaligned matrix columns.",
        ],
    }
    receipt_path = output.with_name(output.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--qat", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--format",
        dest="requested_format",
        required=True,
        choices=sorted(GGML_FORMATS),
    )
    arguments = parser.parse_args()
    print(json.dumps(export_candidate(**vars(arguments)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
