#!/usr/bin/env python3
"""Benchmark a streaming MLX-LM server with synchronized request bursts."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    label: str
    prompt_chars: int
    concurrency: int
    max_tokens: int


def parse_case(value: str) -> Case:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "case must be label:prompt_chars:concurrency:max_tokens"
        )
    label, prompt_text, concurrency_text, tokens_text = parts
    try:
        prompt_chars = int(prompt_text)
        concurrency = int(concurrency_text)
        max_tokens = int(tokens_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case counts must be integers") from error
    if not label or min(prompt_chars, concurrency, max_tokens) < 1:
        raise argparse.ArgumentTypeError("case label and counts must be positive")
    return Case(label, prompt_chars, concurrency, max_tokens)


def parse_token_ids(value: str) -> list[int]:
    if not value:
        return []
    try:
        ids = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "token IDs must be comma-separated integers"
        ) from error
    if min(ids, default=0) < 0:
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return ids


def build_prompt(base_text: str, case: Case, request_id: int, run_id: str) -> str:
    repeats = case.prompt_chars // max(len(base_text), 1) + 1
    prefix = (base_text * repeats)[: case.prompt_chars]
    marker = f"{run_id}-{case.label}-{request_id}"
    return (
        f"{prefix}\n\nRequest marker: {marker}. Continue with a detailed "
        "technical explanation. Do not stop early."
    )


def build_request_body(
    model: str,
    prompt: str,
    max_tokens: int,
    suppressed_eos_ids: list[int],
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logit_bias": {str(token): -100.0 for token in suppressed_eos_ids},
        "stream": True,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def read_stream(
    response: http.client.HTTPResponse,
    request_start: float,
) -> tuple[dict[str, Any], str]:
    headers_s = time.perf_counter() - request_start
    first_token_s: float | None = None
    token_times: list[float] = []
    output_parts: list[str] = []
    finish_reason: str | None = None

    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        choice = event["choices"][0]
        text = choice.get("text", "")
        now = time.perf_counter()
        if text:
            if first_token_s is None:
                first_token_s = now - request_start
            token_times.append(now - request_start)
            output_parts.append(text)
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]

    if first_token_s is None:
        raise RuntimeError("server closed without a non-empty text event")

    done_s = time.perf_counter() - request_start
    decode_s = (
        token_times[-1] - token_times[0]
        if len(token_times) > 1
        else done_s - first_token_s
    )
    latencies = [
        token_times[index] - token_times[index - 1]
        for index in range(1, len(token_times))
    ]
    generated = len(output_parts)
    result = {
        "finish_reason": finish_reason,
        "generated_tokens": generated,
        "request_s": done_s,
        "headers_s": headers_s,
        "first_token_s": first_token_s,
        "decode_tokens_per_s": (generated - 1) / decode_s if generated > 1 else 0,
        "p50_decode_latency_ms": statistics.median(latencies) * 1000
        if latencies
        else 0,
        "p95_decode_latency_ms": percentile(latencies, 0.95) * 1000 if latencies else 0,
    }
    return result, hashlib.sha256("".join(output_parts).encode()).hexdigest()


def one_request(
    host: str,
    port: int,
    body: dict[str, Any],
    barrier: threading.Barrier,
    timeout: float,
    request_id: int,
) -> dict[str, Any]:
    barrier.wait()
    started = time.perf_counter()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        result, output_hash = read_stream(response, started)
    finally:
        connection.close()
    return {"request_id": request_id, **result, "output_sha256": output_hash}


def summarize_group(
    case: Case,
    runs: list[dict[str, Any]],
    group_wall_s: float,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [float(run[field]) for run in runs]

    def value_or_zero(field: str) -> float:
        return statistics.median(values(field)) if runs else 0.0

    total_tokens = sum(int(run["generated_tokens"]) for run in runs)
    return {
        "label": case.label,
        "prompt_chars": case.prompt_chars,
        "concurrency": case.concurrency,
        "max_tokens": case.max_tokens,
        "group_wall_s": group_wall_s,
        "aggregate_completion_tokens_per_s": total_tokens / group_wall_s,
        "sum_per_stream_decode_tokens_per_s": sum(values("decode_tokens_per_s")),
        "total_generated_tokens": total_tokens,
        "median_first_token_s": value_or_zero("first_token_s"),
        "max_first_token_s": max(values("first_token_s"), default=0.0),
        "median_request_s": value_or_zero("request_s"),
        "max_request_s": max(values("request_s"), default=0.0),
        "successful_requests": len(runs),
        "failed_requests": len(errors or []),
        "errors": errors or [],
        "runs": runs,
    }


def run_group(
    host: str,
    port: int,
    model: str,
    base_text: str,
    case: Case,
    suppressed_eos_ids: list[int],
    timeout: float,
    run_id: str,
) -> dict[str, Any]:
    barrier = threading.Barrier(case.concurrency)
    started = time.perf_counter()
    results: list[dict[str, Any] | BaseException] = [None] * case.concurrency

    def worker(index: int) -> None:
        prompt = build_prompt(base_text, case, index, run_id)
        body = build_request_body(model, prompt, case.max_tokens, suppressed_eos_ids)
        try:
            results[index] = one_request(host, port, body, barrier, timeout, index)
        except BaseException as error:
            results[index] = error

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(case.concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = [result for result in results if isinstance(result, BaseException)]
    runs = [result for result in results if isinstance(result, dict)]
    error_rows = [f"{type(error).__name__}: {error}" for error in errors]
    return summarize_group(case, runs, time.perf_counter() - started, error_rows)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        type=parse_case,
        action="append",
        default=None,
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--suppress-eos-tokens",
        type=parse_token_ids,
        default="150001,150003,150005,150018",
        help="Comma-separated token IDs to bias out. Use an empty value to disable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")
    if args.request_timeout <= 0:
        raise SystemExit("request timeout must be positive")
    if args.run_id == "":
        raise SystemExit("run ID must not be empty")

    base_text = args.prompt_file.read_text(encoding="utf-8")
    if not base_text:
        raise SystemExit("prompt file is empty")
    run_id = args.run_id or f"{time.time_ns()}-{args.port}"
    prompt_sha256 = sha256(base_text.encode("utf-8")).hexdigest()
    groups = []
    cases = args.case or [Case("smoke", 80, 1, 8)]
    any_errors = False
    for case in cases:
        group = run_group(
            args.host,
            args.port,
            args.model,
            base_text,
            case,
            args.suppress_eos_tokens,
            args.request_timeout,
            run_id,
        )
        groups.append(group)
        any_errors = any_errors or bool(group["errors"])
        compact = {key: value for key, value in group.items() if key != "runs"}
        print(json.dumps(compact, sort_keys=True), flush=True)

    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "endpoint": f"http://{args.host}:{args.port}/v1/completions",
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": prompt_sha256,
        "cases": [
            {
                "label": case.label,
                "prompt_chars": case.prompt_chars,
                "concurrency": case.concurrency,
                "max_tokens": case.max_tokens,
            }
            for case in cases
        ],
        "request_timeout_s": args.request_timeout,
        "suppressed_eos_tokens": args.suppress_eos_tokens,
        "sampling": "temperature=0, streaming",
        "token_count_method": "non-empty SSE text events",
        "groups": groups,
    }
    write_report(args.output, report)
    print(f"wrote {args.output}", flush=True)
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
