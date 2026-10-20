#!/usr/bin/env python3
"""Validate and benchmark direct DFlash2 weight E2M1 repacking."""

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
    from tools.mxfp4_kernel.dflash2_shapes import LINEARS
    from tools.mxfp4_kernel.mxfp4_matmul import (
        pack_e2m1_bytes,
        repack_e2m1_words_to_bytes,
        unpack_e2m1,
    )
except ModuleNotFoundError:
    from dflash2_shapes import LINEARS
    from mxfp4_matmul import (
        pack_e2m1_bytes,
        repack_e2m1_words_to_bytes,
        unpack_e2m1,
    )


def reference_weight_repack(words: torch.Tensor, columns: int) -> torch.Tensor:
    """Run the original unpack-then-pack weight path."""
    return pack_e2m1_bytes(unpack_e2m1(words, columns))


def reference_weight_repack_bounded(
    words: torch.Tensor, columns: int, block_rows: int = 128
) -> torch.Tensor:
    """Run the reference path in output-row blocks."""
    result = reference_weight_repack(words[:block_rows], columns)
    for start in range(block_rows, words.shape[0], block_rows):
        result = reference_weight_repack(words[start : start + block_rows], columns)
    return result


def compare_weight_repack(
    words: torch.Tensor,
    fast: torch.Tensor,
    columns: int,
    block_rows: int = 128,
) -> bool:
    """Compare direct repacking against aligned reference blocks."""
    equal = True
    for start in range(0, words.shape[0], block_rows):
        stop = min(start + block_rows, words.shape[0])
        reference = reference_weight_repack(words[start:stop], columns)
        equal &= bool(torch.equal(reference, fast[start:stop]))
        del reference
    return equal


def timed_milliseconds(
    operation: Callable[[], object], warmup: int, repeats: int
) -> tuple[float, object]:
    """Return median CUDA event time and the last result."""
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
    outputs: int,
    columns: int,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Check byte equality and latency for one projection weight."""
    words = (
        torch.randint(
            0,
            256,
            (outputs, columns // 8, 4),
            device=device,
            dtype=torch.uint8,
        )
        .view(torch.uint32)
        .reshape(outputs, columns // 8)
    )
    fast = repack_e2m1_words_to_bytes(words)
    equal = compare_weight_repack(words, fast, columns)
    del fast
    reference_ms, reference = timed_milliseconds(
        lambda: reference_weight_repack_bounded(words, columns),
        warmup,
        repeats,
    )
    del reference
    fast_ms, fast_result = timed_milliseconds(
        lambda: repack_e2m1_words_to_bytes(words), warmup, repeats
    )
    del fast_result
    torch.cuda.empty_cache()
    return {
        "name": name,
        "outputs": outputs,
        "columns": columns,
        "reference_ms": reference_ms,
        "reference_mode": "128_row_blocks",
        "fast_ms": fast_ms,
        "speedup": reference_ms / fast_ms,
        "bytes_equal": equal,
        "passed": equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("weight-repack-results.json")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_memory_at_start, _total_memory = torch.cuda.mem_get_info()
    results = []
    for shape in LINEARS:
        result = run_shape(
            shape.name,
            shape.output_features,
            shape.input_features,
            "cuda",
            arguments.warmup,
            arguments.repeats,
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
        "kernel": "Triton direct uint32 E2M1 to byte repack",
        "results": results,
        "non_claims": [
            "Weight repacking is a one-time preparation cost for static projections.",
            "Byte equality does not by itself qualify model quality.",
        ],
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
