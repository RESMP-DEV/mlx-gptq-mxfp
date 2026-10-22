#!/usr/bin/env python3
"""Run a Flash-Next MTP QAT data-scaling curve.

The question this answers is whether the head's held-out gain keeps improving
with more distinct prompts, or whether it saturates.  Each point trains the
same recipe on a different number of banks and measures the same held-out
banks, so the only thing that changes is the amount and diversity of training
data.  A curve that keeps rising answers "add more data"; a curve that
plateaus while the same-corpus number keeps rising answers "the head is
specializing, not memorizing, and more of the same will not help".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-banks", required=True, type=Path, action="append")
    parser.add_argument("--eval-banks", required=True, type=Path, action="append")
    parser.add_argument(
        "--sizes",
        required=True,
        type=int,
        action="append",
        help="How many training banks each point uses.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--confidence-floor", type=float, default=0.0)
    parser.add_argument("--run-name", default="flashnext-mtp-scaling")
    return parser


def run_point(arguments: argparse.Namespace, size: int, index: int) -> dict[str, Any]:
    output = arguments.output / f"point-{index:02d}-banks-{size:03d}"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mlx_gptq.flashnext_mtp_train",
        "--checkpoint",
        str(arguments.repo / "source"),
        "--output",
        str(output),
        "--device",
        arguments.device,
        "--dtype",
        arguments.dtype,
        "--attention",
        arguments.attention,
        "--steps",
        str(arguments.steps),
        "--learning-rate",
        str(arguments.learning_rate),
        "--qat-mxfp4",
        "--wandb-name",
        f"{arguments.run_name}-banks{size:03d}",
    ]
    for bank in arguments.train_banks[:size]:
        command += ["--bank", str(bank)]
    if arguments.confidence_floor > 0.0:
        command += ["--confidence-floor", str(arguments.confidence_floor)]
    started = time.time()
    completed = subprocess.run(
        command, cwd=arguments.repo, capture_output=True, text=True, check=False
    )
    elapsed = time.time() - started
    (output / "train.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "train.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"training failed for {size} banks: {completed.stderr[-2000:]}"
        )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    return {
        "banks": size,
        "train_banks": [str(bank) for bank in arguments.train_banks[:size]],
        "elapsed_seconds": elapsed,
        "steps": receipt["steps"],
        "training_pairs": receipt["training_pairs"],
        "validation_pairs": receipt["validation_pairs"],
        "same_corpus_validation": receipt["final_validation"],
        "initial_validation": receipt["initial_validation"],
        "overflow_count": receipt["overflow_count"],
        "wandb_run_id": receipt.get("wandb", {}).get("id"),
        "artifacts": {
            "lora_sha256": _sha256(output / "flashnext_mtp_lora.safetensors"),
            "qat_sha256": _sha256(output / "flashnext_mtp_mxfp4_qat.safetensors"),
        },
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for index, size in enumerate(sorted(arguments.sizes)):
        points.append(run_point(arguments, size, index))
        (arguments.output / "curve.json").write_text(
            json.dumps({"points": points}, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"points": points}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
