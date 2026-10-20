#!/usr/bin/env python3
"""Validate and benchmark the fused DFlash2 activation MXFP4 packer."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import triton

try:
    from tools.mxfp4_kernel.dflash2_shapes import dominant_shapes
    from tools.mxfp4_kernel.mxfp4_matmul import (
        TILE_M,
        pack_e2m1_bytes,
        prepare_activation_mxfp4_fast,
        preshuffle_scales,
        quantize_activation_mxfp4,
    )
except ModuleNotFoundError:
    from dflash2_shapes import dominant_shapes
    from mxfp4_matmul import (
        TILE_M,
        pack_e2m1_bytes,
        prepare_activation_mxfp4_fast,
        preshuffle_scales,
        quantize_activation_mxfp4,
    )


def reference_activation_pack(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Run the original Python-backed activation preparation path."""
    rows, columns = activations.shape
    padded_rows = ((rows + TILE_M - 1) // TILE_M) * TILE_M
    padded = torch.zeros(
        (padded_rows, columns), device=activations.device, dtype=torch.float32
    )
    padded[:rows] = activations
    codes, scale_bytes = quantize_activation_mxfp4(padded)
    packed = pack_e2m1_bytes(codes.reshape(padded_rows, columns))
    shuffled = preshuffle_scales(scale_bytes.reshape(padded_rows, columns // 32))
    return packed, shuffled, padded_rows, columns


def reference_activation_pack_bounded(
    activations: torch.Tensor, block_rows: int = TILE_M
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Run the reference path in row blocks without retaining large temporaries."""
    result = reference_activation_pack(activations[:block_rows])
    for start in range(block_rows, activations.shape[0], block_rows):
        result = reference_activation_pack(activations[start : start + block_rows])
    return result


def compare_activation_packs(
    activations: torch.Tensor,
    fast: tuple[torch.Tensor, torch.Tensor, int, int],
    block_rows: int = TILE_M,
) -> tuple[bool, bool]:
    """Compare the fused path against aligned reference blocks."""
    codes_equal = True
    scales_equal = True
    for start in range(0, activations.shape[0], block_rows):
        stop = min(start + block_rows, activations.shape[0])
        reference = reference_activation_pack(activations[start:stop])
        block = start // block_rows
        codes_equal &= bool(
            torch.equal(reference[0][: stop - start], fast[0][start:stop])
        )
        scales_equal &= bool(torch.equal(reference[1], fast[1][block].unsqueeze(0)))
        del reference
    return codes_equal, scales_equal


def timed_milliseconds(
    operation: Callable[[], object], warmup: int, repeats: int
) -> tuple[float, object]:
    """Return median CUDA event time in milliseconds and last result."""
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), result


def run_shape(
    name: str,
    rows: int,
    outputs: int,
    columns: int,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Compare byte-for-byte equality and preparation latency for one shape."""
    del outputs
    activations = torch.randn(rows, columns, device=device) * 0.5
    fast = prepare_activation_mxfp4_fast(activations)
    codes_equal, scales_equal = compare_activation_packs(activations, fast)
    del fast

    reference_mode = "128_row_blocks"
    reference_ms, reference = timed_milliseconds(
        lambda: reference_activation_pack_bounded(activations), warmup, repeats
    )
    del reference
    fast_ms, fast = timed_milliseconds(
        lambda: prepare_activation_mxfp4_fast(activations), warmup, repeats
    )
    del fast
    torch.cuda.empty_cache()
    return {
        "name": name,
        "rows": rows,
        "columns": columns,
        "reference_ms": reference_ms,
        "reference_mode": reference_mode,
        "fast_ms": fast_ms,
        "speedup": reference_ms / fast_ms,
        "codes_equal": codes_equal,
        "scales_equal": scales_equal,
        "passed": codes_equal and scales_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("activation-quantizer-results.json")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_memory_at_start, _total_memory = torch.cuda.mem_get_info()
    shapes = dominant_shapes()
    if arguments.rows:
        selected = set(arguments.rows)
        shapes = [shape for shape in shapes if shape[1] in selected]

    results = []
    for name, rows, _outputs, columns in shapes:
        result = run_shape(
            name, rows, _outputs, columns, "cuda", arguments.warmup, arguments.repeats
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "operator": "Codex",
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "free_memory_at_start": free_memory_at_start,
        },
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "kernel": "fused Triton activation max plus quantize and pack",
        "results": results,
        "non_claims": [
            "Byte-equal operands do not by themselves qualify model quality.",
            "Preparation latency does not predict end-to-end throughput.",
            "This benchmark covers activation preparation only.",
        ],
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    failed = [result["name"] for result in results if not result["passed"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
