import json

import pytest
from tools.build_fable_coding_prompts import (
    choose_sessions,
    collect_rows,
    final_user_message,
    parse_quotas,
)


def test_parse_quotas_rejects_nonpositive_quota():
    with pytest.raises(ValueError, match="quota must be positive"):
        parse_quotas("codex:0")


def test_final_user_message_uses_last_user_turn():
    messages = json.dumps(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "new"},
        ]
    )
    assert final_user_message(messages) == "new"


def test_choose_sessions_is_deterministic_and_reserves_extra():
    sessions = {
        "codex": {f"session-{index}" for index in range(12)},
        "swe": {f"swe-{index}" for index in range(5)},
    }
    quotas = {"codex": 2, "swe": 2}

    first = choose_sessions(sessions, quotas, reserve=3)
    second = choose_sessions(sessions, quotas, reserve=3)

    assert first == second
    assert len(first["codex"]) == 6
    assert len(first["swe"]) == 5


def test_choose_sessions_reports_insufficient_sessions():
    with pytest.raises(ValueError, match="has 1 sessions, needs 2"):
        choose_sessions({"codex": {"one"}}, {"codex": 2}, reserve=2)


def test_collect_rows_deduplicates_and_keeps_longest_session_prompt(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    source = {
        "source": ["codex", "codex", "codex", "swe"],
        "session_id": ["a", "a", "b", "c"],
        "messages": [
            '[{"role":"user","content":"short"}]',
            '[{"role":"user","content":"longer prompt"}]',
            '[{"role":"user","content":"duplicate"}]',
            '[{"role":"user","content":"different prompt"}]',
        ],
    }
    path = tmp_path / "train-data-00001.parquet"
    pyarrow.parquet.write_table(pyarrow.Table.from_pydict(source), path)

    rows, counts = collect_rows(
        [path],
        {"codex": ["a", "b"], "swe": ["c"]},
        {"codex": 2, "swe": 1},
        minimum_chars=3,
        maximum_chars=100,
    )

    assert counts == {"codex": 2, "swe": 1}
    prompts = {row["session_id"]: row["prompt"] for row in rows}
    assert prompts["a"] == "longer prompt"
    assert len({row["prompt_sha256"] for row in rows}) == 3
