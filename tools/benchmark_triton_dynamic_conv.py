#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from mlx_gptq.triton_dynamic_conv import triton_grouped_dynamic_convolve


def _eager(source, dynamic, base):
    batch, sequence_length, hidden_size = source.shape
    groups = dynamic.shape[-2]
    group_size = hidden_size // groups
    grouped = source.view(batch, sequence_length, groups, group_size)
    shifted = torch.zeros_like(grouped)
    shifted[:, 1:] = grouped[:, :-1]
    return (
        grouped * (base[0].view(groups, group_size) + dynamic[:, :, 0])
        + shifted * (base[1].view(groups, group_size) + dynamic[:, :, 1])
    ).view_as(source)


def _measure(function, source, dynamic, base, warmup: int, iterations: int):
    for _ in range(warmup):
        result = function(source, dynamic, base)
        result.backward(result) if source.requires_grad else None
    source.grad = dynamic.grad = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        result = function(source, dynamic, base)
        if source.requires_grad:
            result.backward(result)
            source.grad = dynamic.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iterations, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dflash-package", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.dflash_package is not None:
        sys.path.insert(0, str(args.dflash_package))
        from dflash.model import _grouped_dynamic_convolve

        def eager(source, dynamic, base):
            return _grouped_dynamic_convolve(source, dynamic, base, args.group_size)
    else:
        eager = _eager

    torch.manual_seed(31)
    groups = args.hidden_size // args.group_size
    rows = []
    for sequence_length in args.sequence_lengths:
        common = {
            "device": args.device,
            "dtype": torch.bfloat16,
            "requires_grad": True,
        }
        source = torch.randn(1, sequence_length, args.hidden_size, **common)
        dynamic = torch.randn(
            1,
            sequence_length,
            args.kernel_size,
            groups,
            1,
            **common,
        )
        base = torch.randn(
            args.kernel_size,
            args.hidden_size,
            device=args.device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            eager_output = eager(source, dynamic, base).float()
            triton_output = triton_grouped_dynamic_convolve(
                source, dynamic, base, args.group_size
            ).float()
        max_error = (eager_output - triton_output).abs().max().item()
        row = {
            "sequence_length": sequence_length,
            "max_abs_error": max_error,
            "eager_seconds": None,
            "triton_seconds": None,
            "speedup": None,
        }
        if not args.profile:
            eager_time, _ = _measure(
                eager, source, dynamic, base, args.warmup, args.iterations
            )
            triton_time, _ = _measure(
                lambda source, dynamic, base: triton_grouped_dynamic_convolve(
                    source, dynamic, base, args.group_size
                ),
                source,
                dynamic,
                base,
                args.warmup,
                args.iterations,
            )
            row["eager_seconds"] = eager_time
            row["triton_seconds"] = triton_time
            row["speedup"] = eager_time / triton_time
        rows.append(row)
        print(json.dumps(row), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
