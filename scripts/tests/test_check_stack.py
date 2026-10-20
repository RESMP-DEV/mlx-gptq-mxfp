from __future__ import annotations

import json

import scripts.check_stack as check_stack


def test_receipt_aggregates_runner_results(tmp_path):
    calls = []

    def runner(command, cwd, log_path):
        calls.append(command)
        log_path.write_text("log\n")
        return (1 if len(calls) == 5 else 0, 0.1, str(log_path))

    root = tmp_path / "stack"
    (root / "scripts").mkdir(parents=True)
    original = check_stack.source_state
    check_stack.source_state = lambda _: {
        "source_revision": "test",
        "source_dirty": False,
    }
    try:
        receipt = check_stack.write_receipt(runner, root)
    finally:
        check_stack.source_state = original
    assert receipt["passed"] is False
    assert [entry["exit_code"] for entry in receipt["checks"]] == [0, 0, 0, 0, 1]
    saved = json.loads((root / "build/reports/quality/receipt.json").read_text())
    assert saved == receipt
