#!/usr/bin/env python3
"""Measure an isolated local27 target-only batch serving lane.

This is a post-deploy acceptance tool for the target-only continuous-batch
lane. Beyond the throughput matrix it records:

- timed ``/metrics`` samples taken while the matrix runs, sampled on a
  dedicated background thread so request workers are never blocked;
- a SHA-256 hash of the exact visible content of every response, so
  identical-content claims rest on bytes rather than on string equality
  inside this process;
- an optional streaming health request that exercises the SSE path;
- optional token-ID capture, taken only from fields the pinned
  MLX-LM 0.31.3 server actually returns.

Pinned MLX-LM 0.31.3 response contract, verified against ``mlx_lm/server.py``
of the ``mlx-lm==0.31.3`` wheel:

- A non-stream chat completion returns generated token IDs only inside
  ``choices[0].logprobs.content[*]["id"]``, and only when the request body
  carries ``"logprobs": true``. Such a request must not set
  ``top_logprobs``, because a non-zero ``top_logprobs`` replaces that
  structure with the per-position top-token structure instead.
- The streaming response is SSE: ``data: {json}`` frames terminated by
  ``data: [DONE]``. Streamed chunks carry no token IDs, and ``usage``
  appears only when ``stream_options.include_usage`` is requested.

Anything this tool cannot read from those fields is reported as absent
rather than synthesized.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_MODEL = "/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4"
DEFAULT_PROMPT = (
    "Write the integers from 1 to 100, separated by commas. Output only the list."
)

# Bumped when the receipt shape changes in a non-additive way.
RECEIPT_SCHEMA_VERSION = 2

# Socket errors that a retry can recover from on a burst against the stdlib
# HTTP server: EPIPE, ECONNRESET, and EINVAL on a reused connection.
_TRANSIENT_ERRNOS = frozenset({32, 54, 22})


def fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return value


def content_hash(content: Optional[str]) -> Optional[str]:
    """Return the SHA-256 of the exact visible content bytes."""
    if content is None:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def token_ids_hash(token_ids: list[int]) -> str:
    """Return a stable digest of a generated token-ID sequence."""
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def extract_token_ids(payload: dict[str, Any]) -> list[int]:
    """Extract generated token IDs from a pinned MLX-LM 0.31.3 response.

    The pinned server returns token IDs only in
    ``choices[0].logprobs.content[*]["id"]`` for a non-stream response
    requested with ``logprobs: true``. No other field carries token IDs, so
    any other shape yields an empty list.

    The pinned server builds that structure in two mutually exclusive
    branches. The ``logprobs: true`` branch emits ``dict(id=..., logprob=...)``
    entries, while the ``top_logprobs`` branch emits entries that also carry
    ``token`` and ``top_logprobs`` keys, whose ``id`` is a top-logprob
    candidate rather than the generated token. Only the exact
    ``{"id", "logprob"}`` entry shape is therefore read as a generated token
    ID. The probe never requests ``top_logprobs``, so the second branch is
    not taken in practice; the check keeps the reader honest if it ever is.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        return []
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    token_ids: list[int] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        if set(entry) != {"id", "logprob"}:
            continue
        token_id = entry.get("id")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            continue
        token_ids.append(token_id)
    return token_ids


def build_request_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    logprobs: bool = False,
    stream: bool = False,
) -> dict[str, Any]:
    """Build a chat-completion body using only pinned MLX-LM fields."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if logprobs:
        # The pinned server only emits token IDs for a non-stream request
        # with logprobs enabled, and only when top_logprobs is unset.
        body["logprobs"] = True
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _is_transient_socket_error(exc: BaseException) -> bool:
    """Return True for a connection failure that a retry can recover from.

    A burst of concurrent requests against the stdlib HTTP server can be
    refused or reset at the socket level before the request is served. The
    server records those as successful requests, so the failure only shows up
    on the client side. A live 32-way run produced "Broken pipe", "Invalid
    argument", and "Connection reset by peer" this way, and a retry recovered
    every one of them.
    """
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return True
        if isinstance(reason, OSError) and reason.errno in _TRANSIENT_ERRNOS:
            return True
        return False
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
        return True
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS


def completion(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    logprobs: bool = False,
    retries: int = 2,
    retry_backoff_s: float = 0.25,
) -> dict[str, Any]:
    body = json.dumps(
        build_request_body(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
        )
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            break
        except Exception as exc:
            if attempts > retries or not _is_transient_socket_error(exc):
                raise
            time.sleep(retry_backoff_s * attempts)
    wall_s = time.perf_counter() - started
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    content = choice.get("message", {}).get("content")
    row = {
        "wall_s": wall_s,
        "attempts": attempts,
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "content_sha256": content_hash(content),
    }
    if logprobs:
        token_ids = extract_token_ids(payload)
        row["token_ids"] = token_ids
        row["token_ids_sha256"] = token_ids_hash(token_ids) if token_ids else None
        row["token_count"] = len(token_ids)
    return row


def streaming_health(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    """Run one SSE chat-completion request and summarize the streamed result.

    The pinned server streams ``data: {json}`` frames terminated by
    ``data: [DONE]``. Streamed chunks carry no token IDs; the completion
    token count is only available when the request asks for usage.
    """
    body = json.dumps(
        build_request_body(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    chunks = 0
    text_parts: list[str] = []
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = {}
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            chunks += 1
            frame_usage = frame.get("usage")
            if isinstance(frame_usage, dict):
                usage = frame_usage
            for choice in frame.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                reason = choice.get("finish_reason")
                if reason:
                    finish_reason = reason
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        text_parts.append(piece)
    wall_s = time.perf_counter() - started
    content = "".join(text_parts)
    return {
        "wall_s": wall_s,
        "chunks": chunks,
        "done_sentinel": done,
        "finish_reason": finish_reason,
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        ),
        "content": content,
        "content_sha256": content_hash(content),
    }


# Dotted paths of the JSON metrics document that the probe summarises. They
# are the fields the owned DFlash server actually publishes; a path that the
# served document does not carry is omitted rather than invented.
METRICS_SUMMARY_PATHS = (
    "server.version",
    "server.model",
    "server.draft",
    "server.mode",
    "runtime.prefill_step_size",
    "runtime.prefix_cache_enabled",
    "memory.rss_gb",
    "memory.rss_peak_gb",
    "memory.phys_footprint_gb",
    "memory.mlx_active_gb",
    "memory.mlx_cache_gb",
    "memory.mlx_peak_gb",
    "memory.wired_gb",
    "memory.wired_limit_gb",
    "rates.requests_per_s",
    "rates.requests_per_s_60s",
    "rates.generated_tokens_per_s",
    "rates.average_decode_tok_s",
    "rates.prefill_tokens_physical_per_s",
    "rates.prefill_tokens_restored_per_s",
    "rates.active_decode_tok_s",
    "totals.requests",
    "totals.generated_tokens",
    "totals.decode_s",
    "totals.prefill_tokens_physical",
    "totals.prefill_tokens_restored",
    "totals.cache_hits",
    "totals.cache_misses",
    "current_request.request_id",
    "current_request.state",
    "current_request.mode_used",
    "current_request.generated_tokens",
)


def metrics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a metrics document to the stable scalar fields of interest."""
    summary: dict[str, Any] = {}
    for path in METRICS_SUMMARY_PATHS:
        node: Any = payload
        for part in path.split("."):
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if node is None:
            continue
        summary[path] = node
    return summary


class MetricsSampler:
    """Sample the metrics endpoint on a background thread.

    The sampler owns its own thread and its own HTTP fetches, so a slow or
    failing metrics endpoint can never delay or fail a matrix request
    worker.
    """

    def __init__(
        self,
        *,
        fetch: Callable[..., dict[str, Any]],
        url: str,
        interval_s: float,
        timeout: float,
        keep_payloads: bool = False,
    ) -> None:
        self._fetch = fetch
        self._url = url
        self._interval_s = interval_s
        self._timeout = timeout
        self._keep_payloads = keep_payloads
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0

    def _sample_once(self) -> dict[str, Any]:
        elapsed_s = time.perf_counter() - self._started_at
        try:
            payload = self._fetch(url=self._url, timeout=self._timeout)
        except Exception as exc:
            return {
                "elapsed_s": elapsed_s,
                "error": f"{type(exc).__name__}: {exc}",
            }
        sample: dict[str, Any] = {
            "elapsed_s": elapsed_s,
            "summary": metrics_summary(payload),
        }
        if self._keep_payloads:
            sample["metrics"] = payload
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample_once()
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(self._interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name="local27-metrics-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self._interval_s, 1.0))
            self._thread = None
        with self._lock:
            return list(self._samples)

    def samples(self) -> list[dict[str, Any]]:
        """Return the samples collected so far without stopping the sampler."""
        with self._lock:
            return list(self._samples)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)


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
    logprobs: bool = False,
    sampler: Optional[MetricsSampler] = None,
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
                logprobs=logprobs,
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
        row.setdefault("content_sha256", content_hash(row.get("content")))
        return row

    started = time.perf_counter()
    if sampler is not None:
        sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, range(concurrency)))
    finally:
        # The request wall time is measured before the sampler is joined, so
        # the aggregate rate never absorbs sampler shutdown latency.
        wall_s = time.perf_counter() - started
        samples = sampler.stop() if sampler is not None else []
    valid = [row for row in rows if "error" not in row]
    tokens = sum(row["completion_tokens"] for row in valid)
    contents = [row.get("content") for row in valid]
    content_hashes = [row.get("content_sha256") for row in valid]
    token_id_hashes = [row.get("token_ids_sha256") for row in valid]
    distinct_content_hashes = sorted({h for h in content_hashes if h is not None})
    distinct_token_id_hashes = sorted({h for h in token_id_hashes if h is not None})
    content_hashes_captured = bool(content_hashes) and all(
        h is not None for h in content_hashes
    )
    token_ids_captured = bool(token_id_hashes) and all(
        h is not None for h in token_id_hashes
    )
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
        "all_content_hashes_identical": content_hashes_captured
        and len(distinct_content_hashes) == 1,
        "distinct_content_sha256": distinct_content_hashes,
        "all_token_ids_identical": (
            len(distinct_token_id_hashes) == 1 if token_ids_captured else None
        ),
        "distinct_token_ids_sha256": distinct_token_id_hashes,
        "token_ids_captured": token_ids_captured,
        "content_hashes_captured": content_hashes_captured,
        "metrics_samples": len(samples),
        "metrics_sample_errors": sum(1 for s in samples if "error" in s),
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
        "--metrics-interval-ms",
        type=float,
        default=0.0,
        help=(
            "Sampling period for the background /metrics sampler. "
            "0 disables sampling, which is the default because a live "
            "32-way run showed that periodic sampling on the same "
            "server caused client socket errors."
        ),
    )
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=10.0,
        help="Timeout of one background /metrics fetch.",
    )
    parser.add_argument(
        "--keep-metrics-payloads",
        action="store_true",
        help="Record the full metrics document for every sample.",
    )
    parser.add_argument(
        "--stream-health",
        action="store_true",
        help="Send one streaming (SSE) request and record its result.",
    )
    parser.add_argument(
        "--stream-health-max-tokens",
        type=int,
        default=32,
        help="max_tokens of the optional streaming health request.",
    )
    parser.add_argument(
        "--token-ids",
        action="store_true",
        help=(
            "Request logprobs so the pinned server returns generated token "
            "IDs in choices[0].logprobs.content[*].id."
        ),
    )
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
    if args.metrics_interval_ms < 0:
        raise SystemExit("metrics-interval-ms must not be negative")
    if args.stream_health_max_tokens <= 0:
        raise SystemExit("stream-health-max-tokens must be positive")
    metrics_url = args.metrics_url or args.url.rstrip("/") + "/metrics"
    report: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": args.url,
        "model": args.model,
        "token_ids_requested": bool(args.token_ids),
        "metrics_before": fetch_json(metrics_url, timeout=min(args.timeout, 10.0)),
    }
    if args.health_only:
        report["health"] = completion(
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=min(args.max_tokens, 32),
            timeout=args.timeout,
            logprobs=bool(args.token_ids),
        )
        passed = bool(report["health"].get("finish_reason"))
    else:
        sampler: Optional[MetricsSampler] = None
        if args.metrics_interval_ms > 0:
            sampler = MetricsSampler(
                fetch=fetch_json,
                url=metrics_url,
                interval_s=args.metrics_interval_ms / 1000.0,
                timeout=min(args.metrics_timeout, args.timeout),
                keep_payloads=bool(args.keep_metrics_payloads),
            )
        report["matrix"] = run_matrix(
            fetch=completion,
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            stagger_s=args.stagger_ms / 1000.0,
            logprobs=bool(args.token_ids),
            sampler=sampler,
        )
        matrix = report["matrix"]
        # The sampler was stopped by run_matrix, so its samples are final.
        samples = sampler.samples() if sampler is not None else []
        report["metrics_samples"] = samples
        report["metrics_samples_summary"] = [
            sample.get("summary", {"error": sample.get("error")}) for sample in samples
        ]
        passed = (
            matrix["all_finish"]
            and matrix["all_content_identical"]
            and not matrix["errors"]
        )
    if args.stream_health:
        try:
            report["stream_health"] = streaming_health(
                url=args.url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.stream_health_max_tokens,
                timeout=args.timeout,
            )
        except Exception as exc:
            report["stream_health"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
        stream = report["stream_health"]
        passed = (
            bool(passed)
            and bool(stream.get("finish_reason"))
            and not stream.get("error")
        )
    report["metrics_after"] = fetch_json(metrics_url, timeout=min(args.timeout, 10.0))
    report["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
