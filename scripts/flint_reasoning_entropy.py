#!/usr/bin/env python3
"""Calibrate a Flint top-20 logprob entropy reasoning cut."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_gptq.reasoning_entropy import (  # noqa: E402
    calibrate,
    evaluate,
    load_tasks,
    summarize,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    positive = (
        args.calibration_limit,
        args.evaluation_limit,
        args.checkpoint_tokens,
        args.probe_tokens,
    )
    if min(positive) <= 0:
        raise ValueError("limits and token budgets must be positive")
    if args.max_tokens < args.checkpoint_tokens:
        raise ValueError("max tokens must be at least checkpoint tokens")

    calibration_tasks = load_tasks(args.tasks, args.calibration_limit, 0)
    evaluation_tasks = load_tasks(
        args.tasks,
        args.evaluation_limit,
        args.calibration_limit,
    )
    calibration, values = calibrate(
        args.base_url,
        args.model,
        calibration_tasks,
        args.assistant_prefix,
        args.checkpoint_tokens,
        args.probe_tokens,
        args.timeout,
    )
    threshold = float(values[args.threshold_method])
    baselines, policies = evaluate(
        args.base_url,
        args.model,
        evaluation_tasks,
        args.assistant_prefix,
        args.checkpoint_tokens,
        args.probe_tokens,
        args.max_tokens,
        args.timeout,
        threshold,
    )
    baseline_summary = summarize(baselines)
    policy_summary = summarize(policies)
    return {
        "schema": "mlx-stack/flint-reasoning-entropy-controller/1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "statistic": "mean normalized top-20 Shannon entropy in bits",
            "checkpoint_tokens": args.checkpoint_tokens,
            "probe_tokens": args.probe_tokens,
            "threshold_method": args.threshold_method,
            "threshold": threshold,
            "max_tokens": args.max_tokens,
        },
        "calibration": {
            "thresholds": values,
            "summary": summarize(calibration),
            "cases": calibration,
        },
        "evaluation": {
            "baseline": {"summary": baseline_summary, "cases": baselines},
            "policy": {"summary": policy_summary, "cases": policies},
            "cuts": sum(row["decision"] == "cut" for row in policies),
            "continues": sum(row["decision"] == "continue" for row in policies),
            "mean_token_delta": policy_summary["mean_tokens"]
            / baseline_summary["mean_tokens"]
            - 1,
            "accuracy_delta": policy_summary["correct"] / len(policies)
            - baseline_summary["correct"] / len(baselines),
        },
        "non_claims": [
            "A missing entropy payload always selects continue; it never cuts blindly.",
            "Cut requests probe after a real model-generated reasoning prefix.",
            "Exact/substring hit rate is workload-specific, not general quality.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-limit", type=int, default=16)
    parser.add_argument("--evaluation-limit", type=int, default=16)
    parser.add_argument("--checkpoint-tokens", type=int, default=64)
    parser.add_argument("--probe-tokens", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--threshold-method",
        choices=("mean_correct", "zero_observed_error"),
        default="mean_correct",
    )
    parser.add_argument("--assistant-prefix", default="<think>\n")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI failure evidence.
        print(
            f"Flint entropy benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    evaluation = dict(receipt["evaluation"])  # type: ignore[arg-type]
    evaluation["threshold"] = receipt["method"]["threshold"]
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
