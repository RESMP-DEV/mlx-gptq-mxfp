"""Build a text-only BF16 Qwen3.8-27B reference from the vision-language base.

The pinned base is `Qwen3_5ForConditionalGeneration` (vision tower + MTP drafter +
`model.language_model.*` namespacing). For a fair vLLM text-generation comparison
against the quantized text-only artifacts, we flatten to `Qwen3_5ForCausalLM`,
remap language keys, and drop vision/MTP tensors. The tokenizer is restored
byte-for-byte from the base. No quantization is applied; weights stay BF16.

This mirrors `scripts/convert_mxfp4_text_only.py` minus its packed/scale
assertions, which are quantization-specific and do not hold for a dense base.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class FlattenError(ValueError):
    pass


TEXT_PREFIX = "model.language_model"
ARTIFACT_PREFIX = "model."
TEXT_ARCH = "Qwen3_5ForCausalLM"
TEXT_MODEL_TYPE = "qwen3_5_text"
TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recorded_at() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def tensor_key(source_key: str) -> str | None:
    if source_key == "lm_head.weight":
        return source_key
    prefix = f"{TEXT_PREFIX}."
    if source_key.startswith(prefix):
        return f"{ARTIFACT_PREFIX}{source_key[len(prefix) :]}"
    return None


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise FlattenError("source config has no text_config")
    flattened = dict(text)
    flattened["architectures"] = [TEXT_ARCH]
    flattened["model_type"] = text.get("model_type", TEXT_MODEL_TYPE)
    return flattened


def collect_source_shards(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        return [source / name for name in names]
    single = source / "model.safetensors"
    if single.is_file():
        return [single]
    raise FlattenError("no safetensors shards found in source")


def run_flatten(source: Path, output: Path, receipt: Path, *, shard_gib: float = 4.5) -> dict:
    from safetensors.torch import load_file, save_file

    if output.exists() and any(output.iterdir()):
        raise FlattenError(f"output directory is not empty: {output}")
    config = flatten_config(json.loads((source / "config.json").read_text()))
    shards = collect_source_shards(source)

    text_tensors: dict[str, Any] = {}
    for shard in shards:
        for key, tensor in load_file(str(shard), device="cpu").items():
            target = tensor_key(key)
            if target is not None:
                text_tensors[target] = tensor
    if not text_tensors:
        raise FlattenError("no text tensors matched the language prefix")

    output.mkdir(parents=True, exist_ok=True)
    # Single-file save for the text-only BF16 reference (~50 GB); shard to bound RAM.
    max_bytes = int(shard_gib * 1024**3)
    current: dict[str, Any] = {}
    current_bytes = 0
    shard_index = 0
    weight_map: dict[str, str] = {}
    total_size = 0
    for key in sorted(text_tensors):
        tensor = text_tensors[key]
        nbytes = tensor.numel() * tensor.element_size()
        total_size += nbytes
        if current_bytes + nbytes > max_bytes and current:
            shard_index += 1
            fname = f"model-{shard_index:05d}-of-TOTAL.safetensors"
            save_file(current, str(output / fname), metadata={"format": "pt"})
            for k in current:
                weight_map[k] = fname
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += nbytes
    if current:
        shard_index += 1
        fname = f"model-{shard_index:05d}-of-TOTAL.safetensors"
        save_file(current, str(output / fname), metadata={"format": "pt"})
        for k in current:
            weight_map[k] = fname

    # Rewrite shard names with the true total.
    final_map: dict[str, str] = {}
    renames: dict[str, str] = {}
    for old_name in sorted(set(weight_map.values())):
        new_name = old_name.replace("TOTAL", f"{shard_index:05d}")
        renames[old_name] = new_name
        if new_name != old_name:
            (output / old_name).rename(output / new_name)
    for key, old_name in weight_map.items():
        final_map[key] = renames[old_name]

    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": final_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name in ("generation_config.json",):
        f = source / name
        if f.is_file():
            shutil.copyfile(f, output / name)
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        f = source / name
        if not f.is_file():
            raise FlattenError(f"source tokenizer missing {name}")
        shutil.copyfile(f, output / name)
        tokenizer_hashes[name] = sha256_file(f)

    files = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    receipt_obj = {
        "schema": "mlx-gptq-calibration/bf16-text-only-flatten/1",
        "recorded_at": recorded_at(),
        "source": str(source),
        "output": str(output),
        "architectures": [TEXT_ARCH],
        "tensor_count": len(text_tensors),
        "total_size_bytes": total_size,
        "shard_count": shard_index,
        "tokenizer_files": tokenizer_hashes,
        "output_files": files,
        "non_claims": [
            "This flattens config/key layout only; BF16 weight values are unchanged.",
            "Serves as the matched text-generation reference for quantized variants.",
        ],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n")
    return receipt_obj


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--shard-gib", type=float, default=4.5)
    args = parser.parse_args(argv)
    try:
        out = run_flatten(args.source, args.output, args.receipt, shard_gib=args.shard_gib)
    except (OSError, ValueError) as exc:
        print(f"bf16 flatten failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"tensor_count": out["tensor_count"], "shards": out["shard_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
