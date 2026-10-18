import pytest
from tools.build_fable_dflash_prompts import _parse_quotas, select_candidates


def _candidate(source: str, session: str, prompt: str):
    return {"source": source, "session_id": session, "prompt": prompt}


def test_parse_quotas_rejects_nonpositive_count():
    with pytest.raises(ValueError, match="quota must be positive"):
        _parse_quotas("codex:0")


def test_select_candidates_is_session_grouped_and_deterministic():
    candidates = [
        _candidate("codex-a", "one", "x" * 150),
        _candidate("codex-a", "one", "y" * 150),
        _candidate("codex-a", "two", "z" * 150),
        _candidate("codex-b", "three", "a" * 200),
        _candidate("codex-b", "four", "short"),
    ]
    quotas = {"codex-a": 2, "codex-b": 1}

    first = select_candidates(
        candidates,
        quotas,
        seed=7,
        minimum_chars=100,
        maximum_chars=1000,
    )
    second = select_candidates(
        candidates,
        quotas,
        seed=7,
        minimum_chars=100,
        maximum_chars=1000,
    )

    assert first == second
    assert len(first) == 3
    assert {item["session_id"] for item in first} == {"one", "two", "three"}
    assert "short" not in {item["prompt"] for item in first}


def test_select_candidates_reports_insufficient_sessions():
    with pytest.raises(ValueError, match="has 1 valid sessions, needs 2"):
        select_candidates(
            [_candidate("codex-a", "one", "x" * 150)],
            {"codex-a": 2},
            seed=1,
            minimum_chars=100,
            maximum_chars=1000,
        )
