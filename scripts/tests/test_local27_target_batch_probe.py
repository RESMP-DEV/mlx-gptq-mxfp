from __future__ import annotations

import pytest
from scripts.local27_target_batch_probe import build_parser, main, run_matrix


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
