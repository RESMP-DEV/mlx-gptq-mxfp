#!/usr/bin/env python3
"""Validate a DFlash2 training run's mechanical export invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def validate(args: argparse.Namespace) -> dict[str, object]:
    run = Path(args.run)
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    fidelity = receipt.get("replay_fidelity_cosine")
    if fidelity is None or float(fidelity) < args.fidelity_threshold:
        raise RuntimeError("run is missing the replay fidelity gate")
    if int(receipt.get("steps", -1)) != args.expected_steps:
        raise RuntimeError("run step count does not match expected steps")

    checkpoint = torch.load(
        run / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    current_b = {
        name: tensor
        for name, tensor in checkpoint["current_parameters"].items()
        if name.endswith(".b")
    }
    current_zero = [
        name for name, tensor in current_b.items() if not torch.count_nonzero(tensor)
    ]
    if len(current_b) != 46 or current_zero:
        raise RuntimeError(f"checkpoint LoRA b tensors did not update: {current_zero}")

    with safe_open(str(run / "lora_adapters.safetensors"), framework="pt") as file:
        b_tensors = sorted(key for key in file.keys() if key.endswith(".b"))
        maxima = {
            key: float(file.get_tensor(key).abs().max()) for key in b_tensors
        }
    if len(b_tensors) != 46:
        raise RuntimeError(f"expected 46 LoRA b tensors, found {len(b_tensors)}")
    zero = [key for key, maximum in maxima.items() if maximum == 0.0]
    if zero:
        if args.require_selected_update:
            raise RuntimeError(f"selected LoRA b tensors did not update: {zero}")

    qat_pairs = 0
    if args.qat:
        if not receipt.get("qat_mxfp4"):
            raise RuntimeError("run receipt does not record QAT")
        with safe_open(
            str(run / "qat_mxfp4.safetensors"), framework="pt"
        ) as file:
            qat_pairs = sum(1 for key in file.keys() if key.endswith(".q"))
            missing_scales = [
                key[:-2] + ".scales"
                for key in file.keys()
                if key.endswith(".q") and key[:-2] + ".scales" not in file.keys()
            ]
        if missing_scales:
            raise RuntimeError(f"QAT grids are missing scales: {missing_scales[:3]}")
        if qat_pairs != 47:
            raise RuntimeError(f"expected 47 QAT grid pairs, found {qat_pairs}")

    return {
        "steps": receipt["steps"],
        "replay_fidelity_cosine": fidelity,
        "checkpoint_b_tensors": len(current_b),
        "selected_b_tensors": len(b_tensors) - len(zero),
        "selected_initial": bool(zero),
        "qat_grid_pairs": qat_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--fidelity-threshold", type=float, default=0.97)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--require-selected-update", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments), indent=2))


if __name__ == "__main__":
    main()
