"""Select and render a diverse agentic calibration supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def stable_rank(row: dict, seed: str) -> str:
    metadata = row["metadata"]
    identity = metadata.get("trace_id") or metadata.get("instance_id")
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def fields(row: dict):
    metadata = row["metadata"]
    source = metadata["source_dataset"]
    mix = metadata["cleaning"]["teacher_mix"]
    return source["language"], mix["mode"], metadata["project"]["id"]


def select_diverse(rows: list[dict], count: int, seed: str) -> list[dict]:
    remaining = sorted(rows, key=lambda row: stable_rank(row, seed))
    selected = []
    language_counts = Counter()
    mode_counts = Counter()
    projects = set()

    while remaining and len(selected) < count:
        best_index = None
        best_score = None
        for index, row in enumerate(remaining):
            language, mode, project = fields(row)
            # Strongly prefer unseen projects, then underrepresented languages.
            # Keep exact replay near its natural 30% share without requiring it.
            desired_mode = "exact_replay" if len(selected) % 10 in (0, 3, 7) else "critic_approved"
            score = (
                int(project not in projects),
                -language_counts[language],
                int(mode == desired_mode),
                -mode_counts[mode],
                stable_rank(row, seed),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        row = remaining.pop(best_index)
        language, mode, project = fields(row)
        selected.append(row)
        language_counts[language] += 1
        mode_counts[mode] += 1
        projects.add(project)

    if len(selected) != count:
        raise RuntimeError(f"requested {count} rows but selected {len(selected)}")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--seed", default="agentic-calibration-v1")
    parser.add_argument("--output-text", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args(argv)

    outputs = (args.output_text, args.output_jsonl, args.receipt)
    if any(path.exists() for path in outputs):
        parser.error("output paths must not already exist")

    from transformers import AutoTokenizer

    source_bytes = args.source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    selected = select_diverse(rows, args.rows, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rendered = []
    records = []
    for selection_index, row in enumerate(selected):
        text = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        token_count = len(tokenizer.encode(text))
        rendered.append(text)
        language, mode, project = fields(row)
        metadata = row["metadata"]
        actual_tool_calls = sum(
            len(message.get("tool_calls") or []) for message in row["messages"]
        )
        records.append(
            {
                "selection_index": selection_index,
                "trace_id": metadata["trace_id"],
                "instance_id": metadata["instance_id"],
                "project_id": project,
                "language": language,
                "mode": mode,
                "target_kind": metadata["cleaning"]["action_window"]["target_kind"],
                "messages": len(row["messages"]),
                "tool_calls": actual_tool_calls,
                "source_messages": metadata["counts"]["messages"],
                "source_tool_calls": metadata["counts"]["tool_calls"],
                "rendered_lfm_tokens": token_count,
            }
        )

    text_payload = "\n\n".join(rendered) + "\n"
    jsonl_payload = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        for row in selected
    )
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.write_text(text_payload)
    args.output_jsonl.write_text(jsonl_payload)

    receipt = {
        "algorithm": "diverse-teacher-mix-train-selection-v1",
        "source": {
            "dataset": "Infatoshi/kimi-k3-open-swe-distillation",
            "config": "teacher_mix",
            "split": "train",
            "revision": args.dataset_revision,
            "path": str(args.source.resolve()),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(rows),
        },
        "selection": {
            "seed": args.seed,
            "rows": len(selected),
            "projects": len({record["project_id"] for record in records}),
            "languages": dict(sorted(Counter(record["language"] for record in records).items())),
            "modes": dict(sorted(Counter(record["mode"] for record in records).items())),
            "target_kinds": dict(sorted(Counter(record["target_kind"] for record in records).items())),
            "messages": sum(record["messages"] for record in records),
            "tool_calls": sum(record["tool_calls"] for record in records),
            "source_messages": sum(record["source_messages"] for record in records),
            "source_tool_calls": sum(record["source_tool_calls"] for record in records),
            "rendered_lfm_tokens": sum(record["rendered_lfm_tokens"] for record in records),
            "records": records,
        },
        "outputs": {
            "text": {
                "path": str(args.output_text.resolve()),
                "bytes": len(text_payload.encode()),
                "sha256": hashlib.sha256(text_payload.encode()).hexdigest(),
            },
            "jsonl": {
                "path": str(args.output_jsonl.resolve()),
                "bytes": len(jsonl_payload.encode()),
                "sha256": hashlib.sha256(jsonl_payload.encode()).hexdigest(),
            },
        },
        "validation_split_used": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt["selection"], "records": "omitted"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
