from __future__ import annotations

import json
import time
import urllib.error

import pytest
from scripts.local27_target_batch_probe import (
    MetricsSampler,
    build_parser,
    build_request_body,
    completion,
    content_hash,
    extract_token_ids,
    main,
    metrics_summary,
    run_matrix,
    streaming_health,
    token_ids_hash,
)


def _row(index: int, **overrides):
    value = {
        "wall_s": 1.0 + index * 0.01,
        "completion_tokens": 2,
        "prompt_tokens": 3,
        "cached_tokens": 0,
        "finish_reason": "length",
        "content": "same",
    }
    value.update(overrides)
    return value


class _FakeResponse:
    """Minimal stand-in for an urllib HTTP response."""

    def __init__(self, payload=None, lines=()):
        self._payload = payload
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, *_args):
        return json.dumps(self._payload).encode()

    def __iter__(self):
        return iter(self._lines)


def _metrics_payload(requests: int = 3) -> dict:
    return {
        "server": {
            "version": "1.2.3",
            "model": "m",
            "draft": None,
            "mode": "target_batch",
        },
        "runtime": {"prefill_step_size": 512},
        "memory": {"mlx_active_gb": 21.5, "mlx_cache_gb": 8.5},
        "rates": {"requests_per_s": 0.5},
        "totals": {"requests": requests, "generated_tokens": 128},
        "current_request": {"request_id": 7, "state": "decoding"},
    }


def test_matrix_summarizes_identical_rows() -> None:
    result = run_matrix(
        fetch=lambda **_kwargs: _row(0),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=3,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["requests"] == 3
    assert result["completion_tokens"] == 6
    assert result["all_finish"] is True
    assert result["all_content_identical"] is True
    assert result["errors"] == []
    assert result["aggregate_completion_tok_s"] > 0


def test_matrix_reports_backend_error_without_failing_summary() -> None:
    def fetch(**kwargs):
        if kwargs["prompt"] == "prompt":
            raise RuntimeError("backend unavailable")
        return _row(0)

    result = run_matrix(
        fetch=fetch,
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=1,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["all_finish"] is False
    assert result["all_content_identical"] is False
    assert len(result["errors"]) == 1
    assert "RuntimeError" in result["errors"][0]["error"]


def test_parser_accepts_health_mode_and_output() -> None:
    args = build_parser().parse_args(
        ["--health-only", "--output", "build/reports/probe.json"]
    )
    assert args.health_only is True
    assert args.concurrency == 32


def test_matrix_rejects_invalid_concurrency() -> None:
    with pytest.raises(SystemExit, match="positive"):
        main(["--output", "x", "--concurrency", "0"])


def test_matrix_records_exact_content_hash_per_request() -> None:
    result = run_matrix(
        fetch=lambda **_kwargs: _row(0),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    expected = content_hash("same")
    assert result["distinct_content_sha256"] == [expected]
    assert result["all_content_hashes_identical"] is True
    assert all(row["content_sha256"] == expected for row in result["rows"])


def test_matrix_content_hash_detects_visible_difference() -> None:
    contents = iter(["alpha", "beta"])

    def fetch(**_kwargs):
        return _row(0, content=next(contents))

    result = run_matrix(
        fetch=fetch,
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["all_content_identical"] is False
    assert result["all_content_hashes_identical"] is False
    assert len(result["distinct_content_sha256"]) == 2


def test_matrix_reports_absent_token_ids_when_not_requested() -> None:
    result = run_matrix(
        fetch=lambda **_kwargs: _row(0),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["token_ids_captured"] is False
    assert result["all_token_ids_identical"] is None
    assert result["distinct_token_ids_sha256"] == []


def test_matrix_compares_captured_token_id_hashes() -> None:
    result = run_matrix(
        fetch=lambda **_kwargs: _row(
            0, token_ids=[1, 2, 3], token_ids_sha256=token_ids_hash([1, 2, 3])
        ),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["token_ids_captured"] is True
    assert result["all_token_ids_identical"] is True
    assert result["distinct_token_ids_sha256"] == [token_ids_hash([1, 2, 3])]


def test_matrix_detects_divergent_token_ids() -> None:
    hashes = iter([token_ids_hash([1, 2]), token_ids_hash([1, 3])])

    def fetch(**_kwargs):
        return _row(0, token_ids_sha256=next(hashes))

    result = run_matrix(
        fetch=fetch,
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["all_token_ids_identical"] is False
    assert len(result["distinct_token_ids_sha256"]) == 2


def test_matrix_partial_token_id_capture_is_not_called_identical() -> None:
    hashes = iter([token_ids_hash([1, 2]), None])

    def fetch(**_kwargs):
        return _row(0, token_ids_sha256=next(hashes))

    result = run_matrix(
        fetch=fetch,
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
    )

    assert result["token_ids_captured"] is False
    assert result["all_token_ids_identical"] is None


def test_sampler_records_timed_metrics_samples() -> None:
    calls: list[float] = []

    def fake_fetch(*, url, timeout):
        calls.append(timeout)
        return _metrics_payload()

    sampler = MetricsSampler(
        fetch=fake_fetch,
        url="http://backend/metrics",
        interval_s=0.01,
        timeout=5.0,
    )
    sampler.start()
    time.sleep(0.1)
    samples = sampler.stop()

    assert len(samples) >= 1
    assert all("elapsed_s" in sample for sample in samples)
    assert all("error" not in sample for sample in samples)
    assert samples[0]["summary"]["server.mode"] == "target_batch"
    assert samples[0]["summary"]["totals.requests"] == 3
    assert calls and calls[0] == 5.0


def test_sampler_failure_does_not_raise() -> None:
    def fake_fetch(*, url, timeout):
        raise RuntimeError("metrics down")

    sampler = MetricsSampler(
        fetch=fake_fetch,
        url="http://backend/metrics",
        interval_s=0.01,
        timeout=1.0,
    )
    sampler.start()
    time.sleep(0.05)
    samples = sampler.stop()

    assert samples
    assert all("RuntimeError" in sample["error"] for sample in samples)


def test_matrix_samples_metrics_while_requests_run() -> None:
    sampler = MetricsSampler(
        fetch=lambda *, url, timeout: _metrics_payload(),
        url="http://backend/metrics",
        interval_s=0.01,
        timeout=1.0,
    )

    result = run_matrix(
        fetch=lambda **_kwargs: _row(0),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
        sampler=sampler,
    )

    assert result["metrics_samples"] >= 1
    assert result["metrics_sample_errors"] == 0
    assert result["requests"] == 2
    assert result["all_finish"] is True


def test_matrix_wall_time_excludes_sampler_shutdown_latency() -> None:
    def slow_fetch(*, url, timeout):
        time.sleep(0.4)
        return _metrics_payload()

    sampler = MetricsSampler(
        fetch=slow_fetch,
        url="http://backend/metrics",
        interval_s=0.01,
        timeout=5.0,
    )

    result = run_matrix(
        fetch=lambda **_kwargs: _row(0),
        url="http://backend",
        model="model",
        prompt="prompt",
        concurrency=2,
        max_tokens=2,
        timeout=1.0,
        stagger_s=0.0,
        sampler=sampler,
    )

    # The request wall time must not absorb the 0.4 s metrics fetch.
    assert result["wall_s"] < 0.4
    assert result["metrics_samples"] >= 1


def test_metrics_summary_omits_absent_paths() -> None:
    summary = metrics_summary({"server": {"mode": "target_batch"}})

    assert summary == {"server.mode": "target_batch"}


def test_extract_token_ids_reads_pinned_logprobs_structure() -> None:
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"id": 11, "logprob": -0.5},
                        {"id": 12, "logprob": -0.25},
                    ]
                }
            }
        ]
    }

    assert extract_token_ids(payload) == [11, 12]


def test_extract_token_ids_empty_when_structure_absent() -> None:
    assert extract_token_ids({"choices": [{"message": {"content": "x"}}]}) == []
    assert extract_token_ids({}) == []
    assert extract_token_ids({"choices": []}) == []


def test_extract_token_ids_rejects_top_logprob_candidate_ids() -> None:
    # The pinned server's top_logprobs branch writes an "id" key too, but it
    # is a top-logprob candidate rather than the generated token, and the
    # entry carries extra keys.
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "id": 5,
                            "token": "x",
                            "logprob": -0.1,
                            "top_logprobs": ({"id": 5, "token": "x", "logprob": -0.1},),
                        }
                    ]
                }
            }
        ]
    }

    assert extract_token_ids(payload) == []


def test_request_body_keeps_non_stream_shape_and_adds_logprobs() -> None:
    plain = build_request_body(model="m", prompt="p", max_tokens=8)
    assert plain == {
        "model": "m",
        "messages": [{"role": "user", "content": "p"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }

    with_logprobs = build_request_body(
        model="m", prompt="p", max_tokens=8, logprobs=True
    )
    assert with_logprobs["logprobs"] is True
    assert "top_logprobs" not in with_logprobs
    assert "stream" not in with_logprobs


def test_request_body_for_streaming_asks_for_usage() -> None:
    body = build_request_body(model="m", prompt="p", max_tokens=8, stream=True)

    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_streaming_health_parses_sse_frames(monkeypatch) -> None:
    frames = [
        b'data: {"choices": [{"delta": {"content": "1, "}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "2"}}]}\n\n',
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}],'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(lines=frames),
    )

    result = streaming_health(
        url="http://backend",
        model="model",
        prompt="prompt",
        max_tokens=8,
        timeout=1.0,
    )

    assert result["content"] == "1, 2"
    assert result["content_sha256"] == content_hash("1, 2")
    assert result["chunks"] == 3
    assert result["done_sentinel"] is True
    assert result["finish_reason"] == "stop"
    assert result["completion_tokens"] == 2
    assert result["prompt_tokens"] == 5


def test_streaming_health_parses_pinned_server_chunk_shapes(monkeypatch) -> None:
    # Frame shapes taken from the pinned server's streaming writes
    # (mlx_lm/server.py: content chunks, the final chunk, the usage frame,
    # and the [DONE] sentinel).
    def frame(**kwargs):
        base = {
            "id": "c",
            "object": "chat.completion.chunk",
            "model": "m",
            "created": 0,
            "choices": [{"index": 0, "finish_reason": None, "delta": {}}],
        }
        base.update(kwargs)
        return json.dumps(base)

    def chunk_frame(content):
        return frame(
            choices=[
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"role": "assistant", "content": content},
                }
            ]
        )

    final_frame = frame(
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "delta": {"role": "assistant"},
            }
        ]
    )
    usage_frame = json.dumps(
        {
            "id": "c",
            "object": "chat.completion",
            "model": "m",
            "created": 0,
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
    )
    lines = [
        f"data: {chunk_frame('1, ')}\n\n",
        f"data: {chunk_frame('2, 3')}\n\n",
        f"data: {final_frame}\n\n",
        f"data: {usage_frame}\n\n",
        "data: [DONE]\n\n",
    ]
    frames = [line.encode() for line in lines]

    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(lines=frames),
    )

    result = streaming_health(
        url="http://backend",
        model="model",
        prompt="prompt",
        max_tokens=8,
        timeout=1.0,
    )

    assert result["content"] == "1, 2, 3"
    assert result["finish_reason"] == "length"
    assert result["done_sentinel"] is True
    assert result["chunks"] == 4
    assert result["completion_tokens"] == 3
    assert result["prompt_tokens"] == 5


def test_parser_accepts_new_acceptance_options() -> None:
    args = build_parser().parse_args(
        [
            "--output",
            "build/reports/probe.json",
            "--metrics-interval-ms",
            "500",
            "--stream-health",
            "--token-ids",
            "--keep-metrics-payloads",
        ]
    )

    assert args.metrics_interval_ms == 500.0
    assert args.stream_health is True
    assert args.token_ids is True
    assert args.keep_metrics_payloads is True


def test_main_rejects_negative_metrics_interval() -> None:
    with pytest.raises(SystemExit, match="metrics-interval-ms"):
        main(["--output", "x", "--metrics-interval-ms", "-1"])


def test_transient_socket_errors_are_retried(monkeypatch) -> None:
    calls = {"n": 0}
    payload = {
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def fake_urlopen(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError(BrokenPipeError(32, "Broken pipe"))
        return _FakeResponse(payload)

    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen", fake_urlopen
    )

    row = completion(
        url="http://backend",
        model="model",
        prompt="prompt",
        max_tokens=4,
        timeout=1.0,
    )

    assert calls["n"] == 3
    assert row["attempts"] == 3
    assert row["finish_reason"] == "stop"


def test_non_transient_errors_are_not_retried(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(*_args, **_kwargs):
        calls["n"] += 1
        raise urllib.error.HTTPError("http://backend", 400, "bad", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen", fake_urlopen
    )

    with pytest.raises(urllib.error.HTTPError):
        completion(
            url="http://backend",
            model="model",
            prompt="prompt",
            max_tokens=4,
            timeout=1.0,
        )

    assert calls["n"] == 1


def test_metrics_sampling_is_opt_in() -> None:
    # A live 32-way run showed that periodic sampling on the same server
    # caused client socket errors, so the sampler is off unless asked for.
    args = build_parser().parse_args(["--output", "x"])

    assert args.metrics_interval_ms == 0.0


class _FakeUrlopen:
    """Serve pinned-contract shaped responses over a faked urlopen."""

    def __init__(self, *, with_logprobs: bool = True):
        self.with_logprobs = with_logprobs
        self.request_bodies: list[dict] = []

    def __call__(self, request, timeout=None):
        if isinstance(request, str):
            return _FakeResponse(_metrics_payload())
        body = json.loads(request.data.decode())
        self.request_bodies.append(body)
        if body.get("stream"):
            frames = [
                b'data: {"choices": [{"delta": {"content": "1, 2"}}]}\n\n',
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}],'
                b' "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n',
                b"data: [DONE]\n\n",
            ]
            return _FakeResponse(lines=frames)
        choice = {
            "index": 0,
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "same"},
        }
        payload = {
            "id": "c",
            "object": "chat.completion",
            "model": "m",
            "choices": [choice],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        if self.with_logprobs and body.get("logprobs"):
            choice["logprobs"] = {
                "content": [{"id": 100 + i, "logprob": -0.1} for i in range(4)]
            }
        return _FakeResponse(payload)


def test_main_writes_acceptance_receipt(monkeypatch, tmp_path) -> None:
    fake = _FakeUrlopen()
    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen", fake
    )
    output = tmp_path / "probe.json"

    exit_code = main(
        [
            "--output",
            str(output),
            "--url",
            "http://backend",
            "--concurrency",
            "2",
            "--max-tokens",
            "4",
            "--stagger-ms",
            "0",
            "--metrics-interval-ms",
            "5",
            "--stream-health",
            "--token-ids",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["schema_version"] == 2
    assert report["token_ids_requested"] is True

    matrix = report["matrix"]
    assert matrix["requests"] == 2
    assert matrix["all_content_hashes_identical"] is True
    assert matrix["distinct_content_sha256"] == [content_hash("same")]
    assert matrix["token_ids_captured"] is True
    assert matrix["all_token_ids_identical"] is True
    assert matrix["metrics_samples"] >= 1
    assert matrix["metrics_sample_errors"] == 0

    assert report["metrics_samples"]
    assert report["metrics_samples_summary"][0]["server.mode"] == "target_batch"
    # The samples are recorded once, at the top level of the receipt.
    assert "metrics_sample_log" not in matrix

    stream = report["stream_health"]
    assert stream["finish_reason"] == "stop"
    assert stream["done_sentinel"] is True
    assert stream["content_sha256"] == content_hash("1, 2")
    assert stream["completion_tokens"] == 2

    # Token IDs are only ever requested through the pinned logprobs field.
    assert all(
        body.get("logprobs") is True
        for body in fake.request_bodies
        if "logprobs" in body
    )
    assert all("top_logprobs" not in body for body in fake.request_bodies)


def test_main_health_only_receipt_has_no_matrix(monkeypatch, tmp_path) -> None:
    fake = _FakeUrlopen()
    monkeypatch.setattr(
        "scripts.local27_target_batch_probe.urllib.request.urlopen", fake
    )
    output = tmp_path / "health.json"

    exit_code = main(
        [
            "--output",
            str(output),
            "--url",
            "http://backend",
            "--health-only",
            "--max-tokens",
            "4",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text())
    assert "matrix" not in report
    assert report["health"]["content_sha256"] == content_hash("same")
