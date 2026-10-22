#!/usr/bin/env python3
"""Run all MLX Stack component checks and write one receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "build/reports/quality/receipt.json"


def source_state(root: Path = ROOT) -> dict[str, Any]:
    """Return commit and dirty state without including file names."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {"source_revision": revision, "source_dirty": bool(status)}


def _environment(cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    parts = (str(cwd), str(ROOT), env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(part for part in parts if part)
    return env


def default_runner(
    command: list[str], cwd: Path, log_path: Path
) -> tuple[int, float, str]:
    """Run one check and return exit code, elapsed seconds, and log path."""
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_environment(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode, time.monotonic() - started, str(log_path)


def checks(root: Path = ROOT) -> list[tuple[str, list[str], Path]]:
    """Return the ordered component checks."""
    return [
        (
            "root_lint",
            [
                "uv",
                "run",
                "--frozen",
                "ruff",
                "check",
                "mlx_gptq/flashnext_mtp.py",
                "mlx_gptq/flashnext_mtp_eval.py",
                "mlx_gptq/flashnext_mtp_model.py",
                "mlx_gptq/flashnext_mtp_train.py",
                "mlx_gptq/ggml_native.py",
                "scripts",
                "tools/mxfp4_kernel",
                "tests/test_grid_and_packing.py",
                "tests/test_flashnext_mtp.py",
                "tests/test_ggml_native.py",
                "tests/test_flashnext_mtp_eval.py",
                "tests/test_flashnext_mtp_mixing.py",
                "tests/test_flashnext_mtp_train.py",
                "tests/test_export_flashnext_mtp_checkpoint.py",
                "tests/test_flashnext_mtp.py",
                "tests/test_flashnext_mtp_train.py",
                "tests/test_export_flashnext_mtp_checkpoint.py",
            ],
            root,
        ),
        (
            "root_tests",
            ["uv", "run", "--frozen", "pytest", "tests", "-q"],
            root,
        ),
        (
            "runtime_tests",
            [
                "uv",
                "run",
                "--no-project",
                "--with",
                "mlx==0.32.1",
                "--with",
                "mlx-lm==0.31.3",
                "--with",
                "pytest",
                "pytest",
                "tests",
                "-q",
            ],
            root / "runtime/dflash-mlx",
        ),
        (
            "campaign_tests",
            ["uv", "run", "--frozen", "pytest", "tests", "-q"],
            root / "campaigns/mlx-gptq-calibration",
        ),
        (
            "cuda_compile",
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "-m",
                "py_compile",
                "cuda/mlx-cuda-mxfp/probe_mxfp_cuda.py",
            ],
            root,
        ),
    ]


def write_receipt(
    runner: Callable[[list[str], Path, Path], tuple[int, float, str]],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Run checks and atomically record their aggregate result."""
    receipt_path = root / "build/reports/quality/receipt.json"
    log_dir = receipt_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.unlink(missing_ok=True)
    entries: list[dict[str, Any]] = []
    try:
        for name, command, cwd in checks(root):
            exit_code, elapsed, log_path = runner(command, cwd, log_dir / f"{name}.log")
            entries.append(
                {
                    "name": name,
                    "command": " ".join(command),
                    "cwd": str(cwd.relative_to(root)),
                    "exit_code": exit_code,
                    "elapsed_seconds": elapsed,
                    "log": log_path,
                }
            )
        state = source_state(root)
        failure = "check failure"
    except Exception as exc:
        state = source_state(root)
        failure = f"{type(exc).__name__}: {exc}"
        entries.append(
            {
                "name": "dispatcher",
                "command": "internal",
                "exit_code": 1,
                "error": failure,
            }
        )

    receipt = {
        **state,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "operator": "Codex",
        "mode": "quality",
        "scope": (
            "Root engine, DFlash runtime, calibration campaign, and CUDA probe "
            "checks; root lint covers the active kernel and layout-test surface"
        ),
        "non_claims": [
            "CPU checks do not prove CUDA or Metal kernel performance.",
            "Synthetic tests do not qualify DFlash2 model quality.",
            "No model artifact is uploaded or published by this dispatcher.",
        ],
        "checks": entries,
        "passed": bool(entries)
        and all(entry.get("exit_code") == 0 for entry in entries),
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    receipt = write_receipt(default_runner)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
