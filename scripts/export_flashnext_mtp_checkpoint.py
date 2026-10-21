#!/usr/bin/env python3
"""Export only the native MTP, shared embedding, and shared head tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_checkpoint(source: Path, output: Path) -> dict[str, object]:
    source_index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = source_index["weight_map"]
    selected_names = [
        name
        for name in weight_map
        if name.startswith("mtp.")
        or name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}
    ]
    selected_set = set(selected_names)
    shards = sorted({weight_map[name] for name in selected_names})
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"compact checkpoint output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    exported_map: dict[str, str] = {}
    files: list[dict[str, object]] = []

    for shard in shards:
        tensors = load_file(str(source / shard))
        selected = {
            name: tensors[name] for name in sorted(selected_set) if name in tensors
        }
        if not selected:
            continue
        destination = output / shard
        save_file(selected, str(destination), metadata={"source_shard": shard})
        size = destination.stat().st_size
        files.append(
            {
                "file": shard,
                "bytes": size,
                "sha256": sha256(destination),
                "tensors": sorted(selected),
            }
        )
        for name in selected:
            exported_map[name] = shard
        del tensors, selected

    index = {
        "metadata": source_index.get("metadata", {}),
        "weight_map": dict(sorted(exported_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(source / "config.json", output / "config.json")
    expected = set(selected_names)
    if set(exported_map) != expected:
        raise RuntimeError("selected tensor set changed while exporting checkpoint")

    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "tensor_count": len(expected),
        "shard_count": len(files),
        "total_bytes": sum(cast(int, file["bytes"]) for file in files),
        "files": files,
        "non_claims": [
            "This is a training input artifact, not a runnable Strata MTP "
            "runtime artifact.",
            "The trunk, vision tower, PLE table, and tokenizer are intentionally "
            "omitted.",
        ],
    }
    (output / "export-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(export_checkpoint(arguments.source, arguments.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
