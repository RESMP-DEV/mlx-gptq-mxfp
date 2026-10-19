#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from mlx_gptq.dflash2_calibrate import _load_reference
from mlx_gptq.dflash2_distill import (
    CycleDataset,
    _forward,
    _install_lora,
    _selector_loss,
)
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--dflash-package", type=Path, required=True)
    parser.add_argument("--initial-projection", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cycles", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(43)
    model = _load_reference(args.model, args.dflash_package, args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _install_lora(model, args.rank, args.alpha)
    selector = model.candidate_selector.hidden_projection
    selector.weight = torch.nn.Parameter(
        selector.weight.detach().float(), requires_grad=True
    )
    selector.weight.data.copy_(
        load_file(args.initial_projection, device="cpu")["weight"].to(
            selector.weight.device
        )
    )

    if args.triton:
        from mlx_gptq.triton_dynamic_conv import triton_grouped_dynamic_convolve

        sys.path.insert(0, str(args.dflash_package))
        import dflash.model as dflash_model

        dflash_model._grouped_dynamic_convolve = triton_grouped_dynamic_convolve

    dataset = CycleDataset(args.capture_dir, args.cycles, stratified=True, seed=43)
    payloads = [dataset[index][0] for index in range(min(len(dataset), args.cycles))]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.01)

    def train_one(payload):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = _forward(model, payload, args.device)
            loss = _selector_loss(
                model,
                payload,
                hidden,
                args.device,
                predecessor_mode="predicted",
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for payload in payloads[: args.warmup]:
        train_one(payload)
    measured = payloads[args.warmup :]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for payload in measured:
        train_one(payload)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "triton": args.triton,
        "cycles": len(measured),
        "elapsed_seconds": elapsed,
        "seconds_per_cycle": elapsed / max(1, len(measured)),
        "rank": args.rank,
        "alpha": args.alpha,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
