#!/usr/bin/env python3
"""Measure an isolated local27 target-only batch serving lane."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4"
DEFAULT_PROMPT = (
    "Write the integers from 1 to 100, separated by commas. Output only the list."
)


def fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return value


def completion(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    wall_s = time.perf_counter() - started
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    return {
        "wall_s": wall_s,
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content"),
    }


def run_matrix(
    *,
    fetch: Callable[..., dict[str, Any]],
    url: str,
    model: str,
    prompt: str,
    concurrency: int,
    max_tokens: int,
    timeout: float,
    stagger_s: float,
) -> dict[str, Any]:
    def one(index: int) -> dict[str, Any]:
        time.sleep(stagger_s * index)
        started = time.perf_counter()
        try:
            row = fetch(
                url=url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            return {
                "i": index,
                "queued_s": stagger_s * index,
                "wall_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["i"] = index
        row["queued_s"] = stagger_s * index
        return row

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, range(concurrency)))
    wall_s = time.perf_counter() - started
    valid = [row for row in rows if "error" not in row]
    tokens = sum(row["completion_tokens"] for row in valid)
    contents = [row.get("content") for row in valid]
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "wall_s": wall_s,
        "completion_tokens": tokens,
        "aggregate_completion_tok_s": tokens / wall_s if wall_s else None,
        "median_request_completion_tok_s": (
            statistics.median(
                row["completion_tokens"] / row["wall_s"]
                for row in valid
                if row["wall_s"] > 0
            )
            if valid
            else None
        ),
        "p50_request_wall_s": (
            statistics.median(row["wall_s"] for row in valid) if valid else None
        ),
        "all_finish": bool(valid) and all(row.get("finish_reason") for row in valid),
        "all_content_identical": bool(contents)
        and all(content == contents[0] for content in contents),
        "errors": [row for row in rows if "error" in row],
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--metrics-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--stagger-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Send one request, write the receipt, and skip the matrix.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency <= 0 or args.max_tokens <= 0:
        raise SystemExit("concurrency and max_tokens must be positive")
    metrics_url = args.metrics_url or args.url.rstrip("/") + "/metrics"
    report: dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": args.url,
        "model": args.model,
        "metrics_before": fetch_json(metrics_url, timeout=min(args.timeout, 10.0)),
    }
    if args.health_only:
        report["health"] = completion(
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=min(args.max_tokens, 32),
            timeout=args.timeout,
        )
        passed = bool(report["health"].get("finish_reason"))
    else:
        report["matrix"] = run_matrix(
            fetch=completion,
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stagger_s=args.stagger_ms / 1000.0,
        )
        matrix = report["matrix"]
        passed = (
            matrix["all_finish"]
            and matrix["all_content_identical"]
            and not matrix["errors"]
        )
    report["metrics_after"] = fetch_json(metrics_url, timeout=min(args.timeout, 10.0))
    report["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
