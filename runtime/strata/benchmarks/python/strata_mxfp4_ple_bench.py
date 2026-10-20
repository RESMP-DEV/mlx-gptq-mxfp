# Copyright © 2026 Apple Inc.

import argparse
import fcntl
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

WIDTH = 160
GROUPS = 5
PACKED_WORDS = 20
CODE_BYTES = PACKED_WORDS * 4
SCALE_BYTES = GROUPS
ROW_BYTES = CODE_BYTES + SCALE_BYTES
PAGE_SIZE = os.sysconf("SC_PAGESIZE")


@dataclass
class PageJob:
    plane: str
    offset: int
    length: int
    uses: list[tuple[int, int]]


@dataclass
class Ticket:
    indices_shape: tuple[int, ...]
    code_out: bytearray
    scale_out: bytearray
    jobs: list[PageJob]
    futures: list[Future]
    row_ids: np.ndarray
    pending_codes: dict[int, bytes]
    pending_scales: dict[int, bytes]
    cache_hits: int
    started_ns: int
    issue_seconds: float


@dataclass
class Stats:
    rows: int
    cache_hits: int
    page_reads: int
    bytes_read: int
    issue_seconds: float
    wait_seconds: float
    dequantize_seconds: float
    elapsed_seconds: float


class Mxfp4PleTable:
    """Read native MXFP4 PLE rows from separate codes and scales planes."""

    def __init__(
        self,
        prefix: Path,
        max_inflight: int,
        cache_rows: int,
        max_selected_rows: int,
    ):
        if not 1 <= max_inflight <= 1024:
            raise ValueError("max-inflight must be in 1..1024")
        if cache_rows < 0:
            raise ValueError("cache-rows must be nonnegative")
        if max_selected_rows <= 0:
            raise ValueError("max-selected-rows must be positive")
        self.codes_path = prefix.with_suffix(prefix.suffix + ".codes")
        self.scales_path = prefix.with_suffix(prefix.suffix + ".scales")
        self.code_file = self.codes_path.open("rb")
        self.scale_file = self.scales_path.open("rb")
        for file in (self.code_file, self.scale_file):
            try:
                fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
            except (AttributeError, OSError):
                pass
        code_size = os.fstat(self.code_file.fileno()).st_size
        scale_size = os.fstat(self.scale_file.fileno()).st_size
        self.plane_sizes = {"codes": code_size, "scales": scale_size}
        code_rows, rem = divmod(code_size, CODE_BYTES)
        scale_rows, scale_rem = divmod(scale_size, SCALE_BYTES)
        if rem or scale_rem or code_rows != scale_rows:
            raise ValueError("codes and scales planes do not have matching rows")
        self.rows = code_rows
        if self.rows <= 0:
            raise ValueError("table has no rows")
        self.max_selected_rows = max_selected_rows
        self.executor = ThreadPoolExecutor(max_workers=max_inflight)
        self.cache_lock = threading.Lock()
        self.cache: OrderedDict[int, tuple[bytes, bytes]] | None = (
            OrderedDict() if cache_rows else None
        )
        self.cache_capacity = cache_rows
        self.cache_size = 0

    def _cache_get(self, row: int) -> tuple[bytes, bytes] | None:
        with self.cache_lock:
            if self.cache is None or row not in self.cache:
                return None
            self.cache.move_to_end(row)
            return self.cache[row]

    def _cache_put(self, row: int, value: tuple[bytes, bytes]) -> None:
        with self.cache_lock:
            if self.cache is None:
                return
            if row in self.cache:
                self.cache.move_to_end(row)
                self.cache[row] = value
            elif self.cache_size < self.cache_capacity:
                self.cache[row] = value
                self.cache_size += 1
            elif self.cache:
                self.cache.popitem(last=False)
                self.cache[row] = value

    @staticmethod
    def _add_job(
        jobs: dict[tuple[str, int], PageJob],
        plane: str,
        at: int,
        row_bytes: int,
        position: int,
    ) -> None:
        first = at // PAGE_SIZE * PAGE_SIZE
        last = (at + row_bytes - 1) // PAGE_SIZE
        length = (last + 1) * PAGE_SIZE - first
        job = jobs.get((plane, first))
        if job is None:
            job = PageJob(plane, first, length, [])
            jobs[(plane, first)] = job
        else:
            job.length = max(job.length, length)
        job.uses.append((position, at - first))

    def issue(self, indices: np.ndarray) -> Ticket:
        started = time.perf_counter_ns()
        source = np.asarray(indices)
        if source.ndim == 0:
            raise ValueError("indices must have at least one dimension")
        if source.dtype.kind not in "iu":
            raise TypeError("indices must use an integer dtype")
        if source.dtype.kind == "i" and np.any(source < 0):
            raise IndexError("negative table index")
        flat = source.reshape(-1).astype(np.uint64, copy=True)
        if flat.size > self.max_selected_rows:
            raise ValueError(f"selection exceeds {self.max_selected_rows} rows")
        if np.any(flat >= self.rows):
            raise IndexError(f"table index exceeds {self.rows - 1}")
        code_out = bytearray(flat.size * CODE_BYTES)
        scale_out = bytearray(flat.size * SCALE_BYTES)
        jobs: dict[tuple[str, int], PageJob] = {}
        cache_hits = 0

        for position, row in enumerate(flat.tolist()):
            hit = self._cache_get(row)
            if hit is not None:
                code_out[position * CODE_BYTES : (position + 1) * CODE_BYTES] = hit[0]
                scale_out[position * SCALE_BYTES : (position + 1) * SCALE_BYTES] = hit[
                    1
                ]
                cache_hits += 1
                continue
            self._add_job(jobs, "codes", row * CODE_BYTES, CODE_BYTES, position)
            self._add_job(jobs, "scales", row * SCALE_BYTES, SCALE_BYTES, position)

        ordered = sorted(jobs.values(), key=lambda job: (job.plane, job.offset))
        for job in ordered:
            job.length = min(job.length, self.plane_sizes[job.plane] - job.offset)
        files = {"codes": self.code_file, "scales": self.scale_file}
        futures = [
            self.executor.submit(self._read_page, files[job.plane], job)
            for job in ordered
        ]
        return Ticket(
            indices.shape,
            code_out,
            scale_out,
            ordered,
            futures,
            flat,
            {},
            {},
            cache_hits,
            started,
            (time.perf_counter_ns() - started) / 1e9,
        )

    def collect(self, ticket: Ticket) -> tuple[mx.array, Stats]:
        wait_started = time.perf_counter_ns()
        try:
            for job, future in zip(ticket.jobs, ticket.futures, strict=True):
                page = future.result()
                width = CODE_BYTES if job.plane == "codes" else SCALE_BYTES
                out = ticket.code_out if job.plane == "codes" else ticket.scale_out
                for position, offset in job.uses:
                    value = page[offset : offset + width]
                    out[position * width : (position + 1) * width] = value
                    if job.plane == "codes":
                        ticket.pending_codes[position] = bytes(value)
                    else:
                        ticket.pending_scales[position] = bytes(value)
        except BaseException:
            for future in ticket.futures:
                future.cancel()
            raise

        for position in range(int(np.prod(ticket.indices_shape))):
            if (
                self.cache is not None
                and position in ticket.pending_codes
                and position in ticket.pending_scales
            ):
                self._cache_put(
                    int(ticket.row_ids[position]),
                    (ticket.pending_codes[position], ticket.pending_scales[position]),
                )
        wait_seconds = (time.perf_counter_ns() - wait_started) / 1e9

        dequantize_started = time.perf_counter_ns()
        shape = (*ticket.indices_shape, PACKED_WORDS)
        codes = mx.asarray(
            np.frombuffer(ticket.code_out, dtype=np.uint32).reshape(shape), copy=True
        )
        scales = mx.asarray(
            np.frombuffer(ticket.scale_out, dtype=np.uint8).reshape(
                (*shape[:-1], GROUPS)
            ),
            copy=True,
        )
        values = mx.dequantize(
            codes,
            scales=scales,
            group_size=32,
            bits=4,
            mode="mxfp4",
        ).astype(mx.float32)
        mx.eval(values)
        mx.synchronize()
        dequantize_seconds = (time.perf_counter_ns() - dequantize_started) / 1e9
        stats = Stats(
            rows=int(np.prod(ticket.indices_shape)),
            cache_hits=ticket.cache_hits,
            page_reads=len(ticket.jobs),
            bytes_read=sum(job.length for job in ticket.jobs),
            issue_seconds=ticket.issue_seconds,
            wait_seconds=wait_seconds,
            dequantize_seconds=dequantize_seconds,
            elapsed_seconds=(time.perf_counter_ns() - ticket.started_ns) / 1e9,
        )
        return values, stats

    def gather(self, indices: np.ndarray) -> tuple[mx.array, Stats]:
        return self.collect(self.issue(indices))

    @staticmethod
    def _read_page(file, job: PageJob) -> bytes:
        page = bytearray()
        while len(page) < job.length:
            chunk = os.pread(
                file.fileno(), job.length - len(page), job.offset + len(page)
            )
            if not chunk:
                raise RuntimeError(f"EOF in {job.plane} read at {job.offset}")
            page.extend(chunk)
        return bytes(page)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.code_file.close()
        self.scale_file.close()


def reference_row(row: int) -> mx.array:
    values = (np.arange(WIDTH, dtype=np.float32) * 0.001 + 1.0) * (row % 37 + 1)
    values += np.sin(np.float64(row) + np.arange(WIDTH, dtype=np.float64) * 0.01) * 0.01
    return mx.array(values[None, :], dtype=mx.float32)


def quantize_rows(rows: np.ndarray) -> tuple[mx.array, mx.array]:
    row_ids = np.asarray(rows, dtype=np.uint64)
    values = (np.arange(WIDTH, dtype=np.float32) * np.float32(0.001) + 1.0)[None, :] * (
        (row_ids % 37 + 1)[:, None].astype(np.float32)
    )
    values += (
        np.sin(
            row_ids[:, None].astype(np.float64)
            + np.arange(WIDTH, dtype=np.float64)[None, :] * 0.01
        )
        * 0.01
    ).astype(np.float32)
    return mx.quantize(
        mx.array(values),
        mode="mxfp4",
    )


def prepare(prefix: Path, rows: int, overwrite: bool) -> bool:
    if rows <= 0:
        raise ValueError("rows must be positive")
    codes_path = prefix.with_suffix(prefix.suffix + ".codes")
    scales_path = prefix.with_suffix(prefix.suffix + ".scales")
    expected = (rows * CODE_BYTES, rows * SCALE_BYTES)
    codes_exists = codes_path.exists()
    scales_exists = scales_path.exists()
    if codes_exists != scales_exists and not overwrite:
        raise FileExistsError("refusing to overwrite a partial plane pair")
    if codes_exists and scales_exists:
        actual = (codes_path.stat().st_size, scales_path.stat().st_size)
        if actual == expected:
            return False
        if not overwrite:
            raise FileExistsError(f"{codes_path} exists with an unexpected size")

    with codes_path.open("wb") as codes_file, scales_path.open("wb") as scales_file:
        try:
            for file in (codes_file, scales_file):
                fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            pass
        block_rows = min(rows, 65536)
        for start in range(0, rows, block_rows):
            count = min(block_rows, rows - start)
            codes, scales = quantize_rows(np.arange(start, start + count))
            codes_file.write(np.asarray(codes, copy=False).tobytes())
            scales_file.write(np.asarray(scales, copy=False).tobytes())
        codes_file.flush()
        scales_file.flush()
        os.fsync(codes_file.fileno())
        os.fsync(scales_file.fileno())
    return True


def verify(table: Mxfp4PleTable) -> None:
    probes = np.array([0, table.rows // 2, table.rows - 1], dtype=np.uint64)
    expected_codes, expected_scales = quantize_rows(probes)
    expected = mx.dequantize(
        expected_codes,
        scales=expected_scales,
        group_size=32,
        bits=4,
        mode="mxfp4",
    ).astype(mx.float32)
    observed, _ = table.gather(probes.reshape(3, 1))
    if not mx.allclose(observed.reshape(3, WIDTH), expected):
        raise RuntimeError("MXFP4 selected rows do not match reference quantization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark grouped MXFP4 PLE row reads and dequantization."
    )
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--rows-per-token", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--cache-rows", type=int, default=0)
    parser.add_argument("--max-selected-rows", type=int, default=65536)
    parser.add_argument("--fresh-indices", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("this benchmark requires the Metal backend")
    positive = (args.rows, args.tokens, args.rows_per_token, args.iterations)
    if any(value <= 0 for value in positive) or args.warmup < 0:
        raise ValueError("rows, tokens, rows/token, and iterations must be positive")
    device_info = mx.device_info(mx.gpu)
    active_before = mx.get_active_memory()
    created = args.prepare and prepare(args.prefix, args.rows, args.overwrite)
    table = Mxfp4PleTable(
        args.prefix,
        args.max_inflight,
        args.cache_rows,
        args.max_selected_rows,
    )
    active_after = mx.get_active_memory()

    rng = np.random.default_rng(args.seed)

    def indices() -> np.ndarray:
        return rng.integers(
            0, table.rows, size=(args.tokens, args.rows_per_token)
        ).astype(np.uint64)

    fixed_indices = indices()
    if created or args.prepare:
        verify(table)

    stats: list[Stats] = []
    for _ in range(args.warmup):
        table.gather(indices() if args.fresh_indices else fixed_indices)
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter_ns()
    for _ in range(args.iterations):
        _, run_stats = table.gather(indices() if args.fresh_indices else fixed_indices)
        stats.append(run_stats)
    elapsed = (time.perf_counter_ns() - start) / 1e9

    selected_rows = args.tokens * args.rows_per_token
    result = {
        "prefix": str(args.prefix.resolve()),
        "rows": table.rows,
        "width": WIDTH,
        "mxfp4": {"group_size": 32, "bits": 4, "code_words": PACKED_WORDS},
        "tokens_per_run": args.tokens,
        "rows_per_token": args.rows_per_token,
        "selected_rows_per_run": selected_rows,
        "fresh_indices": args.fresh_indices,
        "max_selected_rows_per_ticket": args.max_selected_rows,
        "metal": {
            "device": device_info["device_name"],
            "max_buffer_length": device_info["max_buffer_length"],
            "active_memory_before_import": active_before,
            "active_memory_after_import": active_after,
            "active_memory_at_end": mx.get_active_memory(),
            "cache_memory_at_end": mx.get_cache_memory(),
            "peak_memory_at_end": mx.get_peak_memory(),
        },
        "read_cache": {
            "rows": args.cache_rows,
            "filled_at_end": table.cache_size,
            "bytes": table.cache_size * ROW_BYTES,
        },
        "timing": {
            "warmup_runs": args.warmup,
            "measured_runs": args.iterations,
            "elapsed_seconds": elapsed,
            "tokens_per_second": args.iterations * args.tokens / elapsed,
            "logical_bytes_per_second": (
                args.iterations * selected_rows * ROW_BYTES / elapsed
            ),
            "storage_bytes_per_second": (
                sum(item.bytes_read for item in stats) / elapsed
            ),
        },
        "read_totals": {
            "cache_hits": sum(item.cache_hits for item in stats),
            "page_reads": sum(item.page_reads for item in stats),
            "bytes_read": sum(item.bytes_read for item in stats),
            "issue_seconds": sum(item.issue_seconds for item in stats),
            "wait_seconds": sum(item.wait_seconds for item in stats),
            "dequantize_seconds": sum(item.dequantize_seconds for item in stats),
        },
        "last_run": asdict(stats[-1]),
        "non_claims": [
            "synthetic rows are not a licensed model or real token trace",
            "F_NOCACHE is not a traced cold-cache proof on macOS",
            "dequantize-then-project is not the future fused PLE kernel",
            "worker count bounds execution; selected rows bound each ticket",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    table.close()


if __name__ == "__main__":
    main()
