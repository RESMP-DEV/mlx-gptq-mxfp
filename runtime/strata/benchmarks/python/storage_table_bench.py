# Copyright © 2026 Apple Inc.

import argparse
import fcntl
import json
import mmap
import os
import platform
import resource
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
DEFAULT_CHUNK_BYTES = 4 * 1024**3
DEFAULT_ROW_BYTES = 90


@dataclass
class Chunk:
    start_row: int
    row_count: int
    mapping: mmap.mmap
    table: mx.array


@dataclass
class GatherPlan:
    chunks: list[Chunk]
    indices: list[mx.array]
    selected_rows: int

    def run(self) -> None:
        outputs = [
            mx.take(chunk.table, indices, axis=0)
            for chunk, indices in zip(self.chunks, self.indices, strict=True)
            if indices.size != 0
        ]
        mx.eval(outputs)
        mx.synchronize()


@dataclass
class PageJob:
    offset: int
    length: int
    uses: list[tuple[int, int, int]]


@dataclass
class PreadStats:
    rows: int
    cache_hits: int
    page_reads: int
    bytes_read: int
    dedup_rows: int
    issue_seconds: float
    wait_seconds: float
    io_seconds: float
    transfer_seconds: float


@dataclass
class PreadTicket:
    out: bytearray
    jobs: list[PageJob]
    futures: list[Future]
    cache_hits: int
    indices_shape: tuple[int, ...]
    started_ns: int
    issue_seconds: float


class PreadTable:
    """Fetch small variable-width rows with grouped page reads."""

    def __init__(self, path: Path, row_bytes: int, max_inflight: int, cache_rows: int):
        if not 1 <= row_bytes <= PAGE_SIZE:
            raise ValueError("pread rows must contain 1..4096 bytes")
        if not 1 <= max_inflight <= 1024:
            raise ValueError("max-inflight must be in 1..1024")

        self.path = path
        self.row_bytes = row_bytes
        self.file_size = path.stat().st_size
        self.row_count = self.file_size // row_bytes
        if self.row_count == 0:
            raise ValueError("table has no complete rows")

        self.file = path.open("rb")
        self.fd = self.file.fileno()
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
            self.uncached = True
        except (AttributeError, OSError):
            self.uncached = False
        self.executor = ThreadPoolExecutor(max_workers=max_inflight)
        self.cache: OrderedDict[int, bytes] | None = (
            OrderedDict() if cache_rows > 0 else None
        )
        self.cache_capacity = cache_rows
        self.cache_size = 0

    def _cache_get(self, row: int) -> bytes | None:
        if self.cache is None:
            return None
        if row not in self.cache:
            return None
        self.cache.move_to_end(row)
        return self.cache[row]

    def _cache_put(self, row: int, value: bytes) -> None:
        if self.cache is None:
            return
        if row in self.cache:
            self.cache.move_to_end(row)
            self.cache[row] = value
            return
        if self.cache_size < self.cache_capacity:
            self.cache[row] = value
            self.cache_size += 1
        elif self.cache:
            self.cache.popitem(last=False)
            self.cache[row] = value

    def issue(self, indices: np.ndarray) -> PreadTicket:
        flat = indices.reshape(-1).astype(np.uint64, copy=False)
        out = bytearray(flat.size * self.row_bytes)
        jobs_by_offset: dict[int, PageJob] = {}
        io_start = time.perf_counter_ns()
        cache_hits = 0

        for position, row in enumerate(flat.tolist()):
            hit = self._cache_get(row)
            if hit is not None:
                start = position * self.row_bytes
                out[start : start + self.row_bytes] = hit
                cache_hits += 1
                continue

            at = row * self.row_bytes
            first_page = at // PAGE_SIZE * PAGE_SIZE
            in_page = at - first_page
            end_page = (at + self.row_bytes - 1) // PAGE_SIZE
            length = (end_page + 1) * PAGE_SIZE - first_page
            job = jobs_by_offset.get(first_page)
            if job is None:
                job = PageJob(first_page, length, [])
                jobs_by_offset[first_page] = job
            else:
                job.length = max(job.length, length)
            job.uses.append((position, row, in_page))

        jobs = sorted(jobs_by_offset.values(), key=lambda job: job.offset)
        for job in jobs:
            job.length = min(job.length, self.file_size - job.offset)
        futures = [
            self.executor.submit(os.pread, self.fd, job.length, job.offset)
            for job in jobs
        ]
        issue_seconds = (time.perf_counter_ns() - io_start) / 1e9
        return PreadTicket(
            out=out,
            jobs=jobs,
            futures=futures,
            cache_hits=cache_hits,
            indices_shape=indices.shape,
            started_ns=io_start,
            issue_seconds=issue_seconds,
        )

    def collect(self, ticket: PreadTicket) -> tuple[mx.array, PreadStats]:
        out = ticket.out
        jobs = ticket.jobs
        futures = ticket.futures
        cache_hits = ticket.cache_hits
        io_start = ticket.started_ns
        rows = int(np.prod(ticket.indices_shape))
        for job, future in zip(jobs, futures, strict=True):
            page = future.result()
            if len(page) != job.length:
                raise RuntimeError(f"short read at offset {job.offset}")
            for position, row, in_page in job.uses:
                value = page[in_page : in_page + self.row_bytes]
                start = position * self.row_bytes
                out[start : start + self.row_bytes] = value
                self._cache_put(row, bytes(value))
        io_seconds = (time.perf_counter_ns() - io_start) / 1e9

        transfer_start = time.perf_counter_ns()
        shape = (*ticket.indices_shape, self.row_bytes)
        staging = np.frombuffer(out, dtype=np.uint8).reshape(shape)
        table = mx.asarray(staging, copy=True)
        mx.eval(table)
        mx.synchronize()
        transfer_seconds = (time.perf_counter_ns() - transfer_start) / 1e9
        stats = PreadStats(
            rows=rows,
            cache_hits=cache_hits,
            page_reads=len(jobs),
            bytes_read=sum(job.length for job in jobs),
            dedup_rows=rows - cache_hits - len(jobs),
            issue_seconds=ticket.issue_seconds,
            wait_seconds=max(0.0, io_seconds - ticket.issue_seconds),
            io_seconds=io_seconds,
            transfer_seconds=transfer_seconds,
        )
        return table, stats

    def gather(self, indices: np.ndarray) -> tuple[mx.array, PreadStats]:
        return self.collect(self.issue(indices))

    def verify_synthetic(self) -> None:
        probes = np.array(
            [0, self.row_count // 2, self.row_count - 1], dtype=np.uint64
        ).reshape(3, 1)
        table, _ = self.gather(probes)
        expected = (probes % 251).astype(np.uint8).reshape(-1, 1)
        observed = np.asarray(table)[..., 0]
        if not np.array_equal(observed, expected):
            raise RuntimeError("synthetic table rows do not match")


class StorageTable:
    def __init__(self, path: Path, row_bytes: int, chunk_bytes: int):
        if row_bytes <= 0 or row_bytes % PAGE_SIZE != 0:
            raise ValueError(f"row-bytes must be a positive {PAGE_SIZE}-byte multiple")

        self.path = path
        self.row_bytes = row_bytes
        self.file_size = path.stat().st_size
        self.row_count = self.file_size // row_bytes
        if self.row_count == 0:
            raise ValueError("table has no complete rows")

        self.chunks: list[Chunk] = []
        with path.open("rb") as file:
            for start in range(0, self.row_count, self._chunk_rows(chunk_bytes)):
                count = min(self._chunk_rows(chunk_bytes), self.row_count - start)
                length = count * row_bytes
                mapping = mmap.mmap(
                    file.fileno(),
                    length,
                    prot=mmap.PROT_READ,
                    flags=mmap.MAP_SHARED,
                    offset=start * row_bytes,
                )
                view = np.frombuffer(mapping, dtype=np.uint8).reshape(count, row_bytes)
                table = mx.asarray(view, copy=False)
                self.chunks.append(Chunk(start, count, mapping, table))

    def _chunk_rows(self, chunk_bytes: int) -> int:
        return max(1, chunk_bytes // self.row_bytes)

    def plan(self, indices: np.ndarray) -> GatherPlan:
        flat = indices.reshape(-1).astype(np.uint64, copy=False)
        routed: list[mx.array] = []
        selected = 0
        for chunk in self.chunks:
            lower = chunk.start_row
            upper = lower + chunk.row_count
            local = (flat[(flat >= lower) & (flat < upper)] - lower).astype(np.uint32)
            routed.append(mx.asarray(local, copy=True))
            selected += local.size
        if selected != flat.size:
            raise RuntimeError("chunk router dropped indices")
        return GatherPlan(self.chunks, routed, selected)

    def verify_synthetic(self) -> None:
        probes = []
        for chunk in self.chunks:
            row = chunk.start_row + chunk.row_count // 2
            probes.append((chunk, chunk.start_row))
            probes.append((chunk, row))
            probes.append((chunk, chunk.start_row + chunk.row_count - 1))
        for chunk, row in probes:
            index = mx.asarray(
                np.asarray([row - chunk.start_row], dtype=np.uint32), copy=True
            )
            output = mx.take(chunk.table, index, axis=0)
            mx.eval(output)
            mx.synchronize()
            expected = row % 251
            observed = int(np.asarray(output).reshape(-1)[0])
            if observed != expected:
                raise RuntimeError(f"row {row} mismatch: {observed} != {expected}")


def prepare(path: Path, rows: int, row_bytes: int, overwrite: bool) -> bool:
    size = rows * row_bytes
    if path.exists():
        if path.stat().st_size == size:
            return False
        if not overwrite:
            raise FileExistsError(f"{path} exists with an unexpected size")

    with path.open("wb" if overwrite or path.exists() else "xb") as file:
        try:
            fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            pass
        block_rows = min(rows, 1_048_576)
        for start in range(0, rows, block_rows):
            count = min(block_rows, rows - start)
            row_ids = np.arange(start, start + count, dtype=np.uint64) % 251
            block = np.repeat(row_ids[:, None], row_bytes, axis=1).astype(np.uint8)
            file.write(block.tobytes())
        file.flush()
        os.fsync(file.fileno())
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark storage-backed table gathers.")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--backend",
        choices=("pread", "metal-mmap"),
        default="pread",
        help="grouped direct reads, or page-aligned unified-memory gather",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--row-bytes", type=int, default=DEFAULT_ROW_BYTES)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument(
        "--cache-rows",
        type=int,
        default=0,
        help="bounded LRU row cache; zero forces storage reads",
    )
    parser.add_argument("--chunk-mib", type=float)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rows-per-token", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--fresh-indices",
        action="store_true",
        help="draw new pread rows for every run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("this benchmark requires the Metal backend")
    if args.backend == "metal-mmap" and args.row_bytes % PAGE_SIZE != 0:
        raise ValueError("metal-mmap requires a page-multiple row width")

    device_info = mx.device_info(mx.gpu)
    chunk_bytes = (
        int(args.chunk_mib * 1024**2)
        if args.chunk_mib is not None
        else min(DEFAULT_CHUNK_BYTES, device_info["max_buffer_length"])
    )
    chunk_bytes = max(PAGE_SIZE, chunk_bytes // PAGE_SIZE * PAGE_SIZE)

    created = args.prepare and prepare(
        args.path, args.rows, args.row_bytes, args.overwrite
    )
    active_before = mx.get_active_memory()
    if args.backend == "pread":
        table = PreadTable(
            args.path,
            args.row_bytes,
            args.max_inflight,
            args.cache_rows,
        )
        indices = np.random.default_rng(args.seed).integers(
            0, table.row_count, size=(args.tokens, args.rows_per_token)
        )
    else:
        table = StorageTable(args.path, args.row_bytes, chunk_bytes)
        indices = np.random.default_rng(args.seed).integers(
            0, table.row_count, size=(args.tokens, args.rows_per_token)
        )
    plan = table.plan(indices) if args.backend == "metal-mmap" else None
    active_after = mx.get_active_memory()

    if created or args.prepare:
        table.verify_synthetic()

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    stats: list[PreadStats] = []
    rng = np.random.default_rng(args.seed)
    start_ns = time.perf_counter_ns()
    for _ in range(args.warmup):
        if args.backend == "pread":
            if args.fresh_indices:
                indices = rng.integers(
                    0, table.row_count, size=(args.tokens, args.rows_per_token)
                )
            _, run_stats = table.gather(indices)
            stats.append(run_stats)
        else:
            plan.run()
    measured_start_ns = time.perf_counter_ns()
    for _ in range(args.iterations):
        if args.backend == "pread":
            if args.fresh_indices:
                indices = rng.integers(
                    0, table.row_count, size=(args.tokens, args.rows_per_token)
                )
            _, run_stats = table.gather(indices)
            stats.append(run_stats)
        else:
            plan.run()
    elapsed_s = (time.perf_counter_ns() - measured_start_ns) / 1e9
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    selected_rows = args.tokens * args.rows_per_token
    bytes_per_run = selected_rows * table.row_bytes
    result = {
        "backend": args.backend,
        "fresh_indices": args.backend == "pread" and args.fresh_indices,
        "path": str(table.path.resolve()),
        "file_bytes": table.file_size,
        "rows": table.row_count,
        "row_bytes": table.row_bytes,
        "tokens_per_run": args.tokens,
        "rows_per_token": args.rows_per_token,
        "selected_rows_per_run": selected_rows,
        "bytes_per_run": bytes_per_run,
        "uncached_file_hint": getattr(table, "uncached", None),
        "chunk_count": len(getattr(table, "chunks", [])),
        "chunk_bytes": chunk_bytes if args.backend == "metal-mmap" else None,
        "metal": {
            "device": device_info["device_name"],
            "max_buffer_length": device_info["max_buffer_length"],
            "active_memory_before_import": active_before,
            "active_memory_after_import": active_after,
        },
        "timing": {
            "warmup_runs": args.warmup,
            "measured_runs": args.iterations,
            "warmup_elapsed_seconds": (measured_start_ns - start_ns) / 1e9,
            "elapsed_seconds": elapsed_s,
            "tokens_per_second": args.iterations * args.tokens / elapsed_s,
            "bytes_per_second": args.iterations * bytes_per_run / elapsed_s,
            "storage_bytes_per_second": (
                sum(run.bytes_read for run in stats[args.warmup :]) / elapsed_s
                if args.backend == "pread"
                else None
            ),
            "mib_per_second": args.iterations * bytes_per_run / elapsed_s / 1024**2,
        },
        "pread": {
            "cache_rows": args.cache_rows,
            "cache_size_at_end": getattr(table, "cache_size", 0),
            "max_inflight": args.max_inflight,
            "last_run": asdict(stats[-1])
            if args.backend == "pread" and stats
            else None,
            "measured_totals": {
                field: sum(getattr(run, field) for run in stats[args.warmup :])
                if args.backend == "pread"
                else None
                for field in ("cache_hits", "page_reads", "bytes_read", "dedup_rows")
            },
            "measured_io_seconds": (
                sum(run.io_seconds for run in stats[args.warmup :])
                if args.backend == "pread"
                else None
            ),
            "measured_issue_seconds": (
                sum(run.issue_seconds for run in stats[args.warmup :])
                if args.backend == "pread"
                else None
            ),
            "measured_wait_seconds": (
                sum(run.wait_seconds for run in stats[args.warmup :])
                if args.backend == "pread"
                else None
            ),
            "measured_transfer_seconds": (
                sum(run.transfer_seconds for run in stats[args.warmup :])
                if args.backend == "pread"
                else None
            ),
        },
        "page_faults": {
            "minor_delta": rusage_after.ru_minflt - rusage_before.ru_minflt,
            "major_delta": rusage_after.ru_majflt - rusage_before.ru_majflt,
        },
        "non_claims": [
            "macOS F_NOCACHE is an unverified approximation of Linux O_DIRECT"
            if args.backend == "pread"
            else "mmap timing is a warm unified-memory control",
            "the synthetic file may already be resident in the OS page cache",
            "raw-table routing does not validate model output quality",
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": mx.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
