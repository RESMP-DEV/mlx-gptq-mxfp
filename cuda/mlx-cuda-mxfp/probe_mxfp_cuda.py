#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("bf16", "mxfp4", "mxfp8"), default="mxfp4"
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    mx.random.seed(42)
    x = mx.random.normal((args.m, args.k)).astype(mx.bfloat16)
    weight = mx.random.normal((args.n, args.k)).astype(mx.bfloat16)
    if args.mode == "bf16":
        packed = scales = None
        mx.eval(x, weight)
    else:
        packed, scales = mx.quantize(weight, group_size=32, mode=args.mode)
        mx.eval(x, packed, scales)

    def matmul() -> mx.array:
        if args.mode == "bf16":
            return x @ weight.T
        return mx.quantized_matmul(
            x,
            packed,
            scales,
            group_size=32,
            mode=args.mode,
        )

    for _ in range(args.warmup):
        mx.eval(matmul())

    started = time.perf_counter()
    for _ in range(args.iterations):
        mx.eval(matmul())
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "device": str(mx.default_device()),
                "mode": args.mode,
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "iterations": args.iterations,
                "elapsed_seconds": elapsed,
                "milliseconds_per_matmul": elapsed * 1000 / args.iterations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
