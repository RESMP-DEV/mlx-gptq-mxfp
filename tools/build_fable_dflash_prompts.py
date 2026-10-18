#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_quotas(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        source, count_text = item.rsplit(":", 1)
        count = int(count_text)
        if count <= 0:
            raise ValueError(f"quota must be positive: {item}")
        result[source.strip()] = count
    return result


def _valid_prompt(candidate: dict[str, Any], minimum: int, maximum: int) -> bool:
    return minimum <= len(candidate["prompt"]) <= maximum


def select_candidates(
    candidates: list[dict[str, Any]],
    quotas: dict[str, int],
    *,
    seed: int,
    minimum_chars: int,
    maximum_chars: int,
) -> list[dict[str, Any]]:
    by_session: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["source"], candidate["session_id"])
        current = by_session.get(key)
        if current is None or (
            _valid_prompt(candidate, minimum_chars, maximum_chars)
            and not _valid_prompt(current, minimum_chars, maximum_chars)
        ):
            by_session[key] = candidate

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in by_session.values():
        if _valid_prompt(candidate, minimum_chars, maximum_chars):
            grouped[candidate["source"]].append(candidate)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source, count in quotas.items():
        available = grouped.get(source, [])
        if len(available) < count:
            raise ValueError(
                f"source {source!r} has {len(available)} valid sessions, needs {count}"
            )
        selected.extend(rng.sample(available, count))

    prompts_seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in selected:
        digest = hashlib.sha256(candidate["prompt"].encode()).hexdigest()
        if digest in prompts_seen:
            continue
        prompts_seen.add(digest)
        unique.append(candidate)
    rng.shuffle(unique)
    return unique


def _read_candidates(paths: list[Path], sources: set[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    result: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=1000,
            columns=["source", "session_id", "messages"],
        ):
            for row in batch.to_pylist():
                if row["source"] not in sources:
                    continue
                messages = json.loads(row["messages"])
                users = [item for item in messages if item.get("role") == "user"]
                if not users:
                    continue
                result.append(
                    {
                        "source": row["source"],
                        "session_id": row["session_id"],
                        "prompt": users[-1].get("content", ""),
                    }
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--quotas", required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--minimum-chars", type=int, default=100)
    parser.add_argument("--maximum-chars", type=int, default=6000)
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()

    quotas = _parse_quotas(args.quotas)
    paths = sorted(args.dataset.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {args.dataset}")
    candidates = _read_candidates(paths, set(quotas))
    selected = select_candidates(
        candidates,
        quotas,
        seed=args.seed,
        minimum_chars=args.minimum_chars,
        maximum_chars=args.maximum_chars,
    )

    rows = []
    for index, candidate in enumerate(selected):
        rows.append(
            {
                "id": f"fable-agentic-{index:04d}",
                "suite": "fable-agentic",
                "prompt": candidate["prompt"],
                "source": candidate["source"],
                "session_id": candidate["session_id"],
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

    lengths = [len(row["prompt"]) for row in rows]
    receipt = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "dataset_revision": args.dataset_revision,
        "parquet_files": [path.name for path in paths],
        "seed": args.seed,
        "quotas": quotas,
        "prompt_count": len(rows),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "session_count": len({row["session_id"] for row in rows}),
        "prompt_chars": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "non_claims": [
            "Prompts are conditioning samples, not a task-quality evaluation set.",
            "One final user message per session does not preserve a full trace.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
