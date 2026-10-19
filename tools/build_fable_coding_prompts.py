#!/usr/bin/env python3
"""Build a large session-grouped coding prompt corpus from Fable traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_quotas(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        source, count_text = item.rsplit(":", 1)
        count = int(count_text)
        if count <= 0:
            raise ValueError(f"quota must be positive: {item}")
        result[source.strip()] = count
    return result


def final_user_message(messages_text: str) -> str:
    messages = json.loads(messages_text)
    users = [item for item in messages if item.get("role") == "user"]
    if not users:
        return ""
    return str(users[-1].get("content", ""))


def choose_sessions(
    sessions_by_source: dict[str, set[str]], quotas: dict[str, int], reserve: int
) -> dict[str, list[str]]:
    rng = random.Random(31)
    selected: dict[str, list[str]] = {}
    for source, quota in quotas.items():
        available = sorted(sessions_by_source.get(source, set()))
        needed = quota * reserve
        if len(available) < quota:
            raise ValueError(
                f"source {source!r} has {len(available)} sessions, needs {quota}"
            )
        selected[source] = rng.sample(available, min(len(available), needed))
    return selected


def _candidate_key(row: dict[str, Any], minimum: int, maximum: int):
    prompt = final_user_message(row["messages"])
    if not minimum <= len(prompt) <= maximum:
        return None
    return prompt


def collect_rows(
    paths: list[Path],
    selected: dict[str, list[str]],
    quotas: dict[str, int],
    *,
    minimum_chars: int,
    maximum_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_pairs: dict[tuple[str, str], None] = {
        (source, session): None
        for source, sessions in selected.items()
        for session in sessions
    }
    best: dict[tuple[str, str], dict[str, Any]] = {}
    seen_hashes: set[str] = set()
    output: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for path in paths:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                group_index, columns=["source", "session_id", "messages"]
            )
            sources = table.column("source").to_pylist()
            sessions = table.column("session_id").to_pylist()
            selected_indices = [
                index
                for index, pair in enumerate(zip(sources, sessions))
                if pair in selected_pairs
            ]
            if not selected_indices:
                continue
            messages = table.column("messages")
            for index in selected_indices:
                pair = (sources[index], sessions[index])
                row = {
                    "source": sources[index],
                    "session_id": sessions[index],
                    "prompt": _candidate_key(
                        {"messages": messages[index].as_py()},
                        minimum_chars,
                        maximum_chars,
                    ),
                }
                if row["prompt"] is None:
                    continue
                current = best.get(pair)
                if current is None or len(row["prompt"]) > len(current["prompt"]):
                    best[pair] = row

    ordered = [
        row for source in quotas for row in best.values() if row["source"] == source
    ]
    for row in ordered:
        source = row["source"]
        if source_counts[source] >= quotas[source]:
            continue
        digest = hashlib.sha256(row["prompt"].encode()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        source_counts[source] += 1
        output.append({**row, "prompt_sha256": digest})
        if all(source_counts[name] >= count for name, count in quotas.items()):
            break

    missing = {
        source: count - source_counts[source]
        for source, count in quotas.items()
        if source_counts[source] < count
    }
    if missing:
        raise ValueError(f"insufficient valid deduplicated prompts: {missing}")
    random.Random(43).shuffle(output)
    return output, dict(sorted(source_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--quotas",
        default=(
            "codex:192,gemini-cli:64,claude-code:32,vscode-insiders:32,"
            "nemotron-swe-v3:96,nemotron-agentic-v1:96,"
            "nemotron-sft-opencode-v1:128,open-swe-traces:128,"
            "opencode-reasoning-2:256"
        ),
    )
    parser.add_argument("--reserve", type=int, default=4)
    parser.add_argument("--minimum-chars", type=int, default=200)
    parser.add_argument("--maximum-chars", type=int, default=8000)
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()

    quotas = parse_quotas(args.quotas)
    paths = sorted(args.dataset.glob("train-data-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no train parquet files under {args.dataset}")

    sessions_by_source: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    for path in paths:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=20000, columns=["source", "session_id"]
        ):
            for row in batch.to_pylist():
                sessions_by_source[row["source"]].add(row["session_id"])
                row_count += 1

    selected = choose_sessions(sessions_by_source, quotas, args.reserve)
    rows, source_counts = collect_rows(
        paths,
        selected,
        quotas,
        minimum_chars=args.minimum_chars,
        maximum_chars=args.maximum_chars,
    )
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "id": f"fable-coding-{index:04d}",
                "suite": "fable-coding",
                "prompt": row["prompt"],
                "source": row["source"],
                "session_id": row["session_id"],
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    lengths = [len(row["prompt"]) for row in output_rows]
    receipt = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "dataset_revision": args.dataset_revision,
        "train_rows_scanned": row_count,
        "train_shards": len(paths),
        "reserve": args.reserve,
        "quotas": quotas,
        "prompt_count": len(output_rows),
        "source_counts": source_counts,
        "session_count": len({row["session_id"] for row in output_rows}),
        "prompt_chars": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "non_claims": [
            "Coding-heavy conditioning prompts are not a task-quality evaluation set.",
            "One final user message per session does not preserve a full trace.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
