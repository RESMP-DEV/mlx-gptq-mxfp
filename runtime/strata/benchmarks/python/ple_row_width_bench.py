#!/usr/bin/env python3
"""Compare page traffic for random PLE row reads at two row widths.

MXFP4 rows are 85 bytes (80 code + 5 scale). BF16 rows are 320 bytes.
Both land inside one 16 KiB page, so the question is whether physical page
traffic per token depends on row width at all. Run the same indices against
both formats to find out instead of assuming the byte ratio carries over.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

WIDTH = 160
GROUPS = 5
PACKED_WORDS = 20
CODE_BYTES = PACKED_WORDS * 4
SCALE_BYTES = GROUPS
MXFP4_ROW_BYTES = CODE_BYTES + SCALE_BYTES
BF16_ROW_BYTES = WIDTH * 2
CHUNK_ROWS = 8192


def plane_paths(prefix: Path, fmt: str) -> dict[str, Path]:
    if fmt == "mxfp4":
        return {
            "codes": prefix.with_suffix(prefix.suffix + ".codes"),
            "scales": prefix.with_suffix(prefix.suffix + ".scales"),
        }
    return {"values": prefix.with_suffix(prefix.suffix + ".bf16")}


def row_bytes(fmt: str) -> dict[str, int]:
    if fmt == "mxfp4":
        return {"codes": CODE_BYTES, "scales": SCALE_BYTES}
    return {"values": BF16_ROW_BYTES}


def prepare(prefix: Path, fmt: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    widths = row_bytes(fmt)
    handles = {
        name: open(path, "wb") for name, path in plane_paths(prefix, fmt).items()
    }
    try:
        for start in range(0, rows, CHUNK_ROWS):
            count = min(CHUNK_ROWS, rows - start)
            for name, handle in handles.items():
                if name == "codes":
                    handle.write(
                        rng.integers(
                            0, 2**32, count * PACKED_WORDS, dtype=np.uint32
                        ).tobytes()
                    )
                elif name == "scales":
                    handle.write(
                        rng.integers(100, 140, count * GROUPS, dtype=np.uint8).tobytes()
                    )
                else:
                    handle.write(
                        rng.standard_normal(count * WIDTH)
                        .astype(np.float16)
                        .view(np.uint16)
                        .tobytes()
                    )
            for name in widths:
                handles[name].flush()
    finally:
        for handle in handles.values():
            handle.close()


class RowTable:
    def __init__(self, prefix: Path, fmt: str, max_inflight: int):
        self.fmt = fmt
        self.widths = row_bytes(fmt)
        self.files = {}
        for name, path in plane_paths(prefix, fmt).items():
            handle = path.open("rb")
            try:
                fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
            except (AttributeError, OSError):
                pass
            self.files[name] = handle
        sizes = {n: os.fstat(f.fileno()).st_size for n, f in self.files.items()}
        counts = [size // self.widths[name] for name, size in sizes.items()]
        if len(set(counts)) != 1 or min(counts) <= 0:
            raise ValueError(f"plane sizes do not describe equal rows: {sizes}")
        self.rows = counts[0]
        self.plane_sizes = sizes
        self.executor = ThreadPoolExecutor(max_workers=max_inflight)

    def _read(self, name: str, row: int) -> bytes:
        handle = self.files[name]
        offset = row * self.widths[name]
        data = os.pread(handle.fileno(), self.widths[name], offset)
        if len(data) != self.widths[name]:
            raise RuntimeError(f"short read in {name} at row {row}")
        return data

    def gather(self, indices: np.ndarray) -> tuple[mx.array, int, int]:
        flat = indices.reshape(-1)
        futures = []
        counters = {name: 0 for name in self.widths}
        for row in flat.tolist():
            for name in self.widths:
                position = counters[name]
                counters[name] += 1
                futures.append(
                    (name, position, self.executor.submit(self._read, name, int(row)))
                )
        payload = {name: [None] * flat.size for name in self.widths}
        for name, position, future in futures:
            payload[name][position] = future.result()
        if self.fmt == "mxfp4":
            codes = np.frombuffer(b"".join(payload["codes"]), dtype=np.uint32)
            scales = np.frombuffer(b"".join(payload["scales"]), dtype=np.uint8)
            values = mx.dequantize(
                mx.array(codes.reshape(flat.size, PACKED_WORDS)),
                scales=mx.array(scales.reshape(flat.size, GROUPS)),
                biases=None,
                bits=4,
                group_size=32,
                mode="mxfp4",
            )
        else:
            raw = np.frombuffer(b"".join(payload["values"]), dtype=np.uint16)
            values = mx.array(raw.view(np.float16).reshape(flat.size, WIDTH)).astype(
                mx.float32
            )
        mx.eval(values)
        reads = len(futures)
        bytes_read = sum(self.widths[name] for name, _, _ in futures)
        return values.reshape(*indices.shape, WIDTH), reads, bytes_read

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        self.executor.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--format", choices=("mxfp4", "bf16"), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--rows", type=int, default=1_048_576)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--rows-per-token", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare(args.prefix, args.format, args.rows, args.seed)

    table = RowTable(args.prefix, args.format, args.max_inflight)
    rng = np.random.default_rng(args.seed)
    batches = [
        rng.integers(0, table.rows, (args.tokens, args.rows_per_token), dtype=np.int64)
        for _ in range(args.warmup + args.iterations)
    ]

    measured = []
    for batch in batches[args.warmup :]:
        started = time.perf_counter()
        lock = threading.Lock()
        totals = {"reads": 0, "bytes": 0}

        def run(token_indices: np.ndarray) -> None:
            _, reads, bytes_read = table.gather(token_indices)
            with lock:
                totals["reads"] += reads
                totals["bytes"] += bytes_read

        threads = [threading.Thread(target=run, args=(row,)) for row in batch]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started
        measured.append((elapsed, totals["reads"], totals["bytes"]))

    total_tokens = args.tokens * len(measured)
    seconds = sum(item[0] for item in measured)
    reads = sum(item[1] for item in measured)
    physical = sum(item[2] for item in measured)

    result = {
        "format": args.format,
        "row_bytes_logical": sum(table.widths.values()),
        "rows": table.rows,
        "tokens": total_tokens,
        "elapsed_seconds": seconds,
        "tokens_per_second": total_tokens / seconds,
        "reads": reads,
        "reads_per_row": reads / (total_tokens * args.rows_per_token),
        "physical_bytes": physical,
        "physical_bytes_per_token": physical / total_tokens,
        "logical_bytes_per_token": args.rows_per_token * sum(table.widths.values()),
        "physical_over_logical": physical
        / (total_tokens * args.rows_per_token * sum(table.widths.values())),
        "page_size": os.sysconf("SC_PAGESIZE"),
    }
    table.close()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
