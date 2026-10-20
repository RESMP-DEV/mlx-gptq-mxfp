from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

STAT_PATTERNS = {
    "candidate_perplexity": r"Mean PPL\(Q\)\s*:\s*([0-9.eE+-]+)",
    "trace_reference_perplexity_clipped": r"Mean PPL\(base\)\s*:\s*([0-9.eE+-]+)",
    "trace_perplexity_ratio_clipped": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.eE+-]+)",
    "mean_kl_divergence": r"Mean\s+KLD:\s*([0-9.eE+-]+)",
    "maximum_kl_divergence": r"Maximum KLD:\s*([0-9.eE+-]+)",
    "p99_9_kl_divergence": r"99\.9%\s+KLD:\s*([0-9.eE+-]+)",
    "p99_kl_divergence": r"99\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "median_kl_divergence": r"Median\s+KLD:\s*([0-9.eE+-]+)",
    "rms_token_probability_delta_percent": r"RMS Δp\s*:\s*([0-9.eE+-]+)",
    "top1_agreement_percent": r"Same top p:\s*([0-9.eE+-]+)",
}

REFERENCE_PPL_PATTERN = r"Final estimate: PPL =\s*([0-9.eE+-]+)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_kl_statistics(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, pattern in STAT_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            values[name] = float(match.group(1))
    return values


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    common = [
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--chunks",
        str(args.chunks),
        "--threads",
        str(args.threads),
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--no-warmup",
    ]
    if getattr(args, "ubatch_size", None) is not None:
        common.extend(["--ubatch-size", str(args.ubatch_size)])
    if getattr(args, "flash_attn", None) is not None:
        common.extend(["--flash-attn", args.flash_attn])
    if getattr(args, "no_kv_offload", False):
        common.append("--no-kv-offload")
    reference = [
        str(args.llama_perplexity),
        "--model",
        str(args.reference),
        "--file",
        str(args.corpus),
        "--save-all-logits",
        str(args.logits_file),
        *common,
    ]
    candidate = [
        str(args.llama_perplexity),
        "--model",
        str(args.candidate),
        "--kl-divergence",
        "--kl-divergence-base",
        str(args.logits_file),
        *common,
    ]
    return reference, candidate


def run_and_log(
    command: list[str], log_path: Path, environment: dict[str, str] | None = None
) -> tuple[str, float]:
    started = time.monotonic()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command, "".join(lines))
    return "".join(lines), elapsed


def file_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matched llama.cpp full-vocabulary BF16-to-quantized logit trace."
    )
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument(
        "--tail-tokens",
        type=int,
        help=(
            "score only this many predictions at the end of each full context; "
            "requires the repository's llama-perplexity tail-logits patch"
        ),
    )
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"))
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument(
        "--reuse-reference-logits",
        action="store_true",
        help="reuse reference-log-probabilities.bin and reference.log in the output directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    for name in ("llama_perplexity", "reference", "candidate", "corpus"):
        path = getattr(args, name)
        if not path.is_file():
            raise SystemExit(f"missing {name.replace('_', ' ')}: {path}")
    if min(args.ctx_size, args.batch_size, args.chunks, args.threads) <= 0:
        raise SystemExit("context, batch, chunk, and thread counts must be positive")
    if args.tail_tokens is not None and not 0 < args.tail_tokens < args.ctx_size:
        raise SystemExit("tail tokens must be positive and smaller than context size")

    if args.ubatch_size is not None:
        args.ubatch_size = min(args.ubatch_size, args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logits_file = args.output_dir / "reference-log-probabilities.bin"
    reference_command, candidate_command = build_commands(args)
    runtime_environment = os.environ.copy()
    if args.tail_tokens is not None:
        runtime_environment["LLAMA_PPL_EVAL_TOKENS"] = str(args.tail_tokens)
    if args.dry_run:
        commands = {
            "environment": {
                "LLAMA_PPL_EVAL_TOKENS": runtime_environment.get("LLAMA_PPL_EVAL_TOKENS")
            },
            "reference": reference_command,
            "candidate": candidate_command,
        }
        print(json.dumps(commands, indent=2))
        return 0

    reference_log = args.output_dir / "reference.log"
    candidate_log = args.output_dir / "candidate.log"
    if args.reuse_reference_logits:
        if not args.logits_file.is_file() or not reference_log.is_file():
            raise SystemExit("cannot reuse reference trace: logits or reference log is missing")
        reference_output = reference_log.read_text()
        reference_seconds = 0.0
    else:
        reference_output, reference_seconds = run_and_log(
            reference_command, reference_log, runtime_environment
        )
    candidate_output, candidate_seconds = run_and_log(
        candidate_command, candidate_log, runtime_environment
    )
    if args.tail_tokens is not None:
        marker = f"evaluating {args.tail_tokens} tail predictions"
        if marker not in reference_output or marker not in candidate_output:
            raise SystemExit(
                "llama-perplexity did not acknowledge the requested tail token count; "
                "apply patches/llama-perplexity-tail-logits.patch"
            )
    statistics = parse_kl_statistics(candidate_output)
    reference_match = re.search(REFERENCE_PPL_PATTERN, reference_output)
    if reference_match:
        reference_perplexity = float(reference_match.group(1))
        statistics["reference_perplexity"] = reference_perplexity
        candidate_perplexity = statistics.get("candidate_perplexity")
        if candidate_perplexity is not None:
            statistics["perplexity_ratio"] = candidate_perplexity / reference_perplexity
    required = {
        "reference_perplexity",
        "candidate_perplexity",
        "mean_kl_divergence",
        "top1_agreement_percent",
    }
    missing = sorted(required - statistics.keys())
    if missing:
        raise SystemExit(f"llama.cpp output omitted required statistics: {missing}")

    receipt = {
        "schema_version": 1,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
        "method": {
            "runtime": "llama.cpp llama-perplexity",
            "comparison": "full-vocabulary KL over a fixed tail of each complete context",
            "reference_storage": "llama.cpp uint16 log-probability trace",
            "reference_storage_caveat": (
                "the trace clips base log probabilities below the llama.cpp -16 floor; "
                "reference perplexity and its ratio are therefore recomputed from the "
                "unclipped reference pass"
            ),
            "ctx_size": args.ctx_size,
            "chunks": args.chunks,
            "tail_tokens": args.tail_tokens,
            "evaluated_tokens": args.chunks
            * (
                args.tail_tokens
                if args.tail_tokens is not None
                else args.ctx_size - 1 - args.ctx_size // 2
            ),
            "batch_size": args.batch_size,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "reference_trace_reused": args.reuse_reference_logits,
        },
        "artifacts": {
            "runtime": file_record(args.llama_perplexity),
            "reference": file_record(args.reference),
            "candidate": file_record(args.candidate),
            "corpus": file_record(args.corpus),
            "reference_logits": file_record(args.logits_file),
        },
        "commands": {
            "reference": reference_command,
            "candidate": candidate_command,
        },
        "elapsed_seconds": {
            "reference": reference_seconds,
            "candidate": candidate_seconds,
        },
        "statistics": statistics,
        "logs": {
            "reference": str(reference_log.resolve()),
            "candidate": str(candidate_log.resolve()),
        },
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "statistics": statistics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
