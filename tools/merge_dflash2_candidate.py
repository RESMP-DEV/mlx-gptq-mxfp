#!/usr/bin/env python3
"""Merge a distilled DFlash2 LoRA candidate into its BF16 base head.

The trainer exports `lora_adapters.safetensors` and
`distilled_hidden_projection.safetensors` relative to the model it was
loaded from. This tool re-applies both at a given blend strength using the
same cast discipline that produced the promoted G3 artifact, so the merged
head is byte-reproducible against that lineage.

Usage::

    python -m tools.merge_dflash2_candidate \
      --base /path/to/DFlash2-G3 \
      --candidate /path/to/run-dir \
      --output /path/to/merged \
      --strength 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SELECTOR = "candidate_selector.hidden_projection.weight"


def _delta(adapters: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return (
        adapters[f"{prefix}.b"].float() @ adapters[f"{prefix}.a"].float()
    ) * float(adapters[f"{prefix}.scaling"].item())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(base: Path, candidate: Path, output: Path, strength: float) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    base_model = base / "model.safetensors"
    adapters_path = candidate / "lora_adapters.safetensors"
    projection_path = candidate / "distilled_hidden_projection.safetensors"
    for path in (base_model, adapters_path, projection_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    weights = load_file(str(base_model), device="cpu")
    adapters = load_file(str(adapters_path), device="cpu")
    projection = load_file(str(projection_path), device="cpu")["weight"]

    prefixes = sorted({key[:-2] for key in adapters if key.endswith(".a")})
    for prefix in prefixes:
        key = f"{prefix}.weight"
        if key not in weights:
            raise KeyError(f"base is missing adapter target {key}")
        delta = (_delta(adapters, prefix) * strength).to(torch.bfloat16)
        weights[key] = (weights[key].float() + delta.float()).to(torch.bfloat16)

    base_projection = weights[SELECTOR].float()
    weights[SELECTOR] = (
        base_projection + (projection.float() - base_projection) * strength
    ).to(torch.bfloat16)

    output.mkdir(parents=True)
    shutil.copy2(base / "config.json", output / "config.json")
    merged_path = output / "model.safetensors"
    save_file(weights, str(merged_path), metadata=None)
    receipt = {
        "schema_version": 1,
        "base": str(base),
        "candidate": str(candidate),
        "output": str(output),
        "strength": strength,
        "adapter_prefixes": len(prefixes),
        "base_model_sha256": _sha256(base_model),
        "adapters_sha256": _sha256(adapters_path),
        "projection_sha256": _sha256(projection_path),
        "merged_model_sha256": _sha256(merged_path),
        "merged_size_bytes": merged_path.stat().st_size,
        "non_claims": [
            "A merged head is a BF16 artifact, not a quantized MLX draft.",
            "Merge success does not prove exact parity or live acceptance.",
        ],
    }
    (output / "blend.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.candidate, args.output, args.strength), indent=2))


if __name__ == "__main__":
    main()
