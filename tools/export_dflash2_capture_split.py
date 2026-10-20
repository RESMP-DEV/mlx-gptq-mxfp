#!/usr/bin/env python3
"""Export session-disjoint DFlash2 capture splits for matched training arms."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def split_prompts(
    prompt_indices: Iterable[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split unique prompt indices exactly as the trainer's random split."""
    unique = sorted(set(prompt_indices))
    if len(unique) < 2:
        raise ValueError("capture split requires at least two prompts")
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    random.Random(seed).shuffle(unique)
    count = max(1, int(len(unique) * fraction))
    if count >= len(unique):
        raise ValueError("validation fraction leaves no training prompts")
    return unique[count:], unique[:count]


def prefix_cycles(records: list[dict[str, Any]], limit: int) -> list[int]:
    """Select contiguous per-prompt cycle prefixes for cache replay."""
    if limit <= 0 or limit >= len(records):
        return list(range(len(records)))
    groups: dict[int, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(int(record["prompt_index"]), []).append(position)
    quotas = {
        prompt: limit * len(positions) // len(records)
        for prompt, positions in groups.items()
    }
    remainder = limit - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda prompt: (
            -(limit * len(groups[prompt]) / len(records) - quotas[prompt]),
            prompt,
        ),
    )
    for prompt in order[:remainder]:
        quotas[prompt] += 1

    selected: set[int] = set()
    for prompt, positions in groups.items():
        selected.update(positions[: quotas[prompt]])
    return sorted(selected)


def _prompt_path(root: Path, prompt_index: int) -> Path:
    return root / "prompts" / f"{prompt_index:06d}.safetensors"


def _cycle_root(root: Path, cycle: int) -> Path:
    return root / "cycles" / f"{cycle:06d}"


def _write_side(
    source: Path,
    destination: Path,
    prompt_records: list[dict[str, Any]],
    cycles: list[dict[str, Any]],
) -> None:
    destination.mkdir(parents=True)
    (destination / "prompts").mkdir()
    (destination / "cycles").mkdir()
    used_prompts = {int(record["prompt_index"]) for record in cycles}
    for record in prompt_records:
        prompt_index = int(record["prompt_index"])
        if prompt_index not in used_prompts:
            continue
        source_path = _prompt_path(source, prompt_index)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, _prompt_path(destination, prompt_index))
    for record in cycles:
        source_cycle = _cycle_root(source, int(record["cycle"]))
        if not source_cycle.is_dir():
            raise FileNotFoundError(source_cycle)
        shutil.copytree(source_cycle, _cycle_root(destination, int(record["cycle"])))
    kept_prompts = [
        record
        for record in prompt_records
        if int(record["prompt_index"]) in used_prompts
    ]
    payload = json.dumps(
        {"prompts": kept_prompts, "cycles": cycles}, separators=(",", ":")
    )
    (destination / "index.json").write_text(payload + "\n", encoding="utf-8")


def export_split(args: argparse.Namespace) -> dict[str, Any]:
    """Create two capture directories whose prompt sets never overlap."""
    source = Path(args.capture_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    index = json.loads((source / "index.json").read_text(encoding="utf-8"))
    prompts = list(index["prompts"])
    cycles = list(index["cycles"])
    prompt_indices = [int(record["prompt_index"]) for record in cycles]
    train_prompts, validation_prompts = split_prompts(
        prompt_indices, args.validation_fraction, args.seed
    )
    if args.train_prompt_limit > 0:
        if args.train_prompt_limit > len(train_prompts):
            raise ValueError("train prompt limit exceeds the training partition")
        train_prompts = train_prompts[: args.train_prompt_limit]
    if args.validation_prompt_limit > 0:
        if args.validation_prompt_limit > len(validation_prompts):
            raise ValueError("validation prompt limit exceeds the partition")
        validation_prompts = validation_prompts[: args.validation_prompt_limit]
    train_set = set(train_prompts)
    validation_set = set(validation_prompts)
    train_records = [
        record for record in cycles if int(record["prompt_index"]) in train_set
    ]
    validation_records = [
        record for record in cycles if int(record["prompt_index"]) in validation_set
    ]
    selected_train = prefix_cycles(train_records, args.train_cycle_limit)
    selected_validation = prefix_cycles(
        validation_records, args.validation_cycle_limit
    )
    selected_train_records = [train_records[index] for index in selected_train]
    selected_validation_records = [
        validation_records[index] for index in selected_validation
    ]

    output.mkdir(parents=True)
    try:
        _write_side(
            source, output / "train", prompts, selected_train_records
        )
        _write_side(
            source,
            output / "validation",
            prompts,
            selected_validation_records,
        )
    except BaseException:
        shutil.rmtree(output)
        raise

    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_capture": str(source),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "partition_train_prompts": len(train_prompts),
        "partition_validation_prompts": len(validation_prompts),
        "selected_train_prompts": len(
            {int(record["prompt_index"]) for record in selected_train_records}
        ),
        "selected_validation_prompts": len(
            {int(record["prompt_index"]) for record in selected_validation_records}
        ),
        "train_cycles": len(selected_train_records),
        "validation_cycles": len(selected_validation_records),
        "non_claims": [
            "This receipt proves prompt disjointness and file presence, not quality.",
            "The split does not validate target logits or runtime acceptance.",
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--train-cycle-limit", type=int, default=0)
    parser.add_argument("--validation-cycle-limit", type=int, default=0)
    parser.add_argument("--train-prompt-limit", type=int, default=0)
    parser.add_argument("--validation-prompt-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    print(json.dumps(export_split(arguments), indent=2))


if __name__ == "__main__":
    main()
