# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

"""Aggregate EAGLE-4 comparison receipts into one serving decision."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _safe(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _target_label(payload: dict[str, Any]) -> str:
    model_ref = str(payload.get("model_ref", ""))
    lowered = model_ref.lower()
    if "mxfp4" in lowered or "4bit" in lowered:
        return "mxfp4"
    if "1d4bf0f" in lowered or "bf16" in lowered:
        return "bf16"
    return Path(model_ref).name or "unknown"


def _block_label(payload: dict[str, Any]) -> str:
    requested = payload.get("requested_block_tokens")
    effective = payload.get("block_tokens")
    if requested is None:
        return f"b{int(effective)}" if effective is not None else "default"
    if effective is None or int(requested) == int(effective):
        return f"b{int(requested)}"
    return f"b{int(requested)}->b{int(effective)}"


def _group_rows(paths: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        key = (
            _target_label(payload),
            str(payload.get("method", "unknown")),
            _block_label(payload),
        )
        grouped.setdefault(key, []).append((path, payload))

    rows: list[dict[str, Any]] = []
    for (target, method, block), receipts in grouped.items():
        tps_values = [_safe(payload.get("generation_tps")) for _, payload in receipts]
        acceptance_values = [
            _safe(payload.get("acceptance_ratio")) for _, payload in receipts
        ]
        match_values = [
            bool(payload.get("exact_native_id_match", False)) for _, payload in receipts
        ]
        last_payload = receipts[-1][1]
        roofline = last_payload.get("roofline") or {}
        rows.append(
            {
                "target": target,
                "method": method,
                "block": block,
                "effective_block_tokens": (
                    int(last_payload.get("block_tokens"))
                    if last_payload.get("block_tokens") is not None
                    else None
                ),
                "runs": len(receipts),
                "generation_tps_median": statistics.median(tps_values),
                "generation_tps_max": max(tps_values),
                "acceptance_median": statistics.median(acceptance_values),
                "exact_match_all": all(match_values),
                "weights_only_roofline_tps": _safe(
                    roofline.get("weights_only_roofline_tps")
                ),
                "receipts": [str(path) for path, _ in receipts],
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["target"], -row["generation_tps_median"], row["method"]),
    )


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if row["method"] != "native" and row["exact_match_all"]
    ]
    if not candidates:
        return []
    best_tps = max(row["generation_tps_median"] for row in candidates)
    return [row for row in candidates if row["generation_tps_median"] == best_tps]


def _markdown(rows: list[dict[str, Any]], best: list[dict[str, Any]]) -> str:
    lines = [
        "# EAGLE-4 serving comparison",
        "",
        "| Target | Method | Block | Runs | TPS med | TPS max | Acc | Exact | Roof |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['method']} | {row['block']} | "
            f"{row['runs']} | {row['generation_tps_median']:.3f} | "
            f"{row['generation_tps_max']:.3f} | {row['acceptance_median']:.3f} | "
            f"{'yes' if row['exact_match_all'] else 'no'} | "
            f"{row['weights_only_roofline_tps']:.3f} |"
        )
    lines.extend(["", "## Best measured speculative serving stack", ""])
    if best:
        for row in best:
            lines.append(
                f"- `{row['target']}` target with `{row['method']}` at "
                f"`{row['block']}`: {row['generation_tps_median']:.3f} tok/s."
            )
    else:
        lines.append("- No speculative receipt is available yet.")
    lines.extend(
        [
            "",
            "The roofline is a weights-only estimate from native TPS and model bytes;",
            "it does not include attention, GDN state, draft compute, or overhead.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [
        path
        for path in sorted(args.matrix_dir.glob("*.json"))
        if path.name != "summary.json" and "ids" not in path.stem
    ]
    rows = _group_rows(paths)
    best = _best_rows(rows)
    payload = {
        "rows": rows,
        "best_measured_speculative": best,
        "non_claims": [
            "Short-prompt measurements do not represent long-context serving.",
            "Exact IDs are a greedy parity check, not a quality benchmark.",
            "Roofline estimates exclude cache, attention, and protocol overhead.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    args.output.with_suffix(".md").write_text(_markdown(rows, best))
    print(json.dumps({"summary": str(args.output) + ".json", "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
