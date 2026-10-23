from __future__ import annotations

import argparse
import json
import sys
from http.server import ThreadingHTTPServer

import pytest
from scripts.flint_mlx_server import main as server_main
from scripts.flint_mlx_server_benchmark import (
    Case,
    build_parser,
    build_prompt,
    build_request_body,
    main,
    parse_case,
    parse_token_ids,
    read_stream,
    summarize_group,
)


class _FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0).encode() if self._lines else b""


def test_parse_case_and_token_ids() -> None:
    case = parse_case("short:80:128:128")
    assert case == Case("short", 80, 128, 128)
    assert parse_token_ids("150001,150003") == [150001, 150003]


def test_parse_case_rejects_bad_shape() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_case("short:80:128")


def test_prompts_are_unique_and_requested_length() -> None:
    case = Case("short", 12, 2, 4)
    prompts = [
        build_prompt("abcdefgh", case, index, "run")
        for index in range(case.concurrency)
    ]
    assert len(set(prompts)) == case.concurrency
    assert all(prompt.startswith("abcdefghabcd") for prompt in prompts)


def test_request_body_contains_stream_and_eos_bias() -> None:
    body = build_request_body("model", "prompt", 128, [150001])
    assert body["stream"] is True
    assert body["logit_bias"] == {"150001": -100.0}


def test_read_stream_counts_non_empty_events() -> None:
    response = _FakeStream(
        [
            'data: {"choices":[{"text":""}]}\n',
            'data: {"choices":[{"text":"A"}]}\n',
            'data: {"choices":[{"text":"B","finish_reason":"length"}]}\n',
            "data: [DONE]\n",
        ]
    )
    result, output_hash = read_stream(response, request_start=0.0)
    assert result["generated_tokens"] == 2
    assert result["finish_reason"] == "length"
    assert output_hash


def test_read_stream_rejects_empty_stream() -> None:
    with pytest.raises(RuntimeError, match="non-empty text event"):
        read_stream(_FakeStream([]), request_start=0.0)


def test_summary_uses_actual_non_empty_event_count() -> None:
    case = Case("short", 80, 2, 128)
    runs = [
        {
            "generated_tokens": 128,
            "decode_tokens_per_s": 6.0,
            "first_token_s": 1.0,
            "request_s": 22.0,
        }
        for _ in range(2)
    ]
    result = summarize_group(case, runs, group_wall_s=20.0)
    assert result["total_generated_tokens"] == 256
    assert result["aggregate_completion_tokens_per_s"] == pytest.approx(12.8)
    assert result["sum_per_stream_decode_tokens_per_s"] == pytest.approx(12.0)
    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 0


def test_group_reports_failure_without_discarding_successes(monkeypatch) -> None:
    from scripts import flint_mlx_server_benchmark as benchmark

    def fake_one_request(*args, **kwargs):
        if args[-1] == 0:
            raise RuntimeError("connection reset")
        return {
            "finish_reason": "length",
            "generated_tokens": 2,
            "request_s": 1.0,
            "headers_s": 0.1,
            "first_token_s": 0.2,
            "decode_tokens_per_s": 2.0,
            "p50_decode_latency_ms": 100.0,
            "p95_decode_latency_ms": 110.0,
            "output_sha256": "0" * 64,
        }

    monkeypatch.setattr(benchmark, "one_request", fake_one_request)
    result = benchmark.run_group(
        host="127.0.0.1",
        port=1,
        model="model",
        base_text="text",
        case=Case("failure", 4, 2, 2),
        suppressed_eos_ids=[],
        timeout=1,
        run_id="run",
    )
    assert result["successful_requests"] == 1
    assert result["failed_requests"] == 1
    assert "RuntimeError: connection reset" in result["errors"]


def test_main_rejects_invalid_port() -> None:
    with pytest.raises(SystemExit, match="port must"):
        main(["--model", "m", "--prompt-file", "x", "--output", "y", "--port", "0"])


def test_server_wrapper_consumes_backlog_and_forwards_args(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_server_main() -> None:
        calls.append(sys.argv.copy())

    monkeypatch.setattr(ThreadingHTTPServer, "request_queue_size", 5)
    code = server_main(
        ["--listen-backlog", "128", "--model", "flint", "--port", "8083"],
        server_main=fake_server_main,
    )
    assert code == 0
    assert calls == [["mlx_lm.server", "--model", "flint", "--port", "8083"]]
    assert ThreadingHTTPServer.request_queue_size == 5


def test_server_wrapper_rejects_zero_backlog(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["server"])
    with pytest.raises(SystemExit):
        server_main(["--listen-backlog", "0"], server_main=lambda: None)


def test_parser_defaults_to_smoke_and_flint_eos_ids() -> None:
    args = build_parser().parse_args(
        ["--model", "m", "--prompt-file", "p", "--output", "o"]
    )
    assert args.case is None
    assert args.suppress_eos_tokens == [150001, 150003, 150005, 150018]


def test_main_writes_provenance_and_failure_receipt(tmp_path, monkeypatch) -> None:
    from scripts import flint_mlx_server_benchmark as benchmark

    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "report.json"
    prompt.write_text("prompt")

    def fake_run_group(*args, **kwargs):
        return {"label": args[4].label, "errors": ["RuntimeError: failed"]}

    monkeypatch.setattr(benchmark, "run_group", fake_run_group)
    code = benchmark.main(
        [
            "--model",
            "model",
            "--prompt-file",
            str(prompt),
            "--output",
            str(output),
            "--run-id",
            "exact-run",
            "--case",
            "smoke:6:1:2",
        ]
    )
    report = json.loads(output.read_text())
    assert code == 1
    assert report["schema_version"] == 1
    assert report["run_id"] == "exact-run"
    assert report["prompt_sha256"]
    assert report["cases"] == [
        {
            "label": "smoke",
            "prompt_chars": 6,
            "concurrency": 1,
            "max_tokens": 2,
        }
    ]
    assert report["token_count_method"] == "non-empty SSE text events"
