#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TextOnlyPackagingError(ValueError):
    pass


def recorded_at() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


TOKENIZER_FILE_NAMES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_config(
    config: Mapping[str, Any], architecture: str, text_model_type: str
) -> dict[str, Any]:
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        msg = "source config has no text_config"
        raise TextOnlyPackagingError(msg)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        msg = "source config has no quantization_config"
        raise TextOnlyPackagingError(msg)
    flattened = dict(text)
    flattened["architectures"] = [architecture]
    flattened["model_type"] = text.get("model_type", text_model_type)
    flattened["quantization_config"] = dict(quantization)
    flattened["quantization_config"]["ignore"] = ["lm_head"]
    return flattened


def tensor_key(source_key: str, checkpoint_prefix: str, artifact_layers_prefix: str) -> str | None:
    if source_key == "lm_head.weight":
        return source_key
    prefix = f"{checkpoint_prefix}."
    if source_key.startswith(prefix):
        return f"{artifact_layers_prefix}{source_key[len(prefix) :]}"
    return None


def verify_source_index(
    index: Mapping[str, Any],
    loaded_keys: set[str],
    checkpoint_prefix: str,
    artifact_layers_prefix: str,
) -> None:
    weight_map = index.get("weight_map", {})
    required = {"model.safetensors", "model_mtp.safetensors"}
    if not required.issubset(set(weight_map.values())):
        msg = "source index does not contain the expected main and MTP shards"
        raise TextOnlyPackagingError(msg)
    if any("model.visual." in key for key in weight_map):
        msg = "text-only source unexpectedly contains vision weights"
        raise TextOnlyPackagingError(msg)
    expected_text_keys = {
        key
        for key in weight_map
        if tensor_key(key, checkpoint_prefix, artifact_layers_prefix) is not None
    }
    loaded_text_keys = {
        key
        for key in loaded_keys
        if tensor_key(key, checkpoint_prefix, artifact_layers_prefix) is not None
    }
    wrong_shard = sorted(
        key for key in expected_text_keys if weight_map.get(key) != "model.safetensors"
    )
    if expected_text_keys != loaded_text_keys or wrong_shard:
        missing = sorted(expected_text_keys - loaded_text_keys)
        unexpected = sorted(loaded_text_keys - expected_text_keys)
        msg = (
            "indexed text tensors do not match loaded model.safetensors keys: "
            f"missing={missing}, unexpected={unexpected}, wrong_shard={wrong_shard}"
        )
        raise TextOnlyPackagingError(msg)


def copy_pinned_tokenizer(
    tokenizer_source: Path, output: Path, verified_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Restore tokenizer files that Transformers may rewrite while saving a model."""
    if set(verified_hashes) != set(TOKENIZER_FILE_NAMES):
        msg = "verified tokenizer hash set is incomplete"
        raise TextOnlyPackagingError(msg)
    for name in TOKENIZER_FILE_NAMES:
        shutil.copyfile(tokenizer_source / name, output / name)
    return dict(verified_hashes)


def verify_pinned_tokenizer(
    tokenizer_source: Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Fail before artifact serialization when tokenizer provenance does not match."""
    if set(expected_hashes) != set(TOKENIZER_FILE_NAMES):
        msg = "campaign does not pin exactly the required tokenizer file set"
        raise TextOnlyPackagingError(msg)
    hashes = {}
    for name in TOKENIZER_FILE_NAMES:
        source_file = tokenizer_source / name
        if not source_file.is_file():
            msg = f"pinned tokenizer source is missing {name}"
            raise TextOnlyPackagingError(msg)
        digest = sha256_file(source_file)
        if digest != expected_hashes[name]:
            msg = f"tokenizer source hash mismatch for {name}: {digest} != {expected_hashes[name]}"
            raise TextOnlyPackagingError(msg)
        hashes[name] = digest
    return hashes


def verify_staged_tokenizer_hashes(files: Mapping[str, str], expected: Mapping[str, str]) -> None:
    mismatches = {
        name: {"expected": expected[name], "staged": files.get(name)}
        for name in TOKENIZER_FILE_NAMES
        if files.get(name) != expected[name]
    }
    if mismatches:
        msg = f"staged tokenizer hashes do not match pinned source: {mismatches}"
        raise TextOnlyPackagingError(msg)


def run_conversion(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file, save_file

    campaign = json.loads(args.campaign.read_text())
    checkpoint_prefix = campaign["model"]["checkpoint_text_prefix"]
    artifact_layers_prefix = campaign["model"]["artifact_text_prefix"]
    output_campaign = campaign["output"]
    output_resolved = args.output.resolve()
    receipt_resolved = args.receipt.resolve()
    if receipt_resolved == output_resolved or receipt_resolved.is_relative_to(output_resolved):
        msg = f"receipt path must not be inside the output directory: {args.receipt}"
        raise TextOnlyPackagingError(msg)
    if args.output.exists() and not args.output.is_dir():
        msg = f"output path is not a directory: {args.output}"
        raise TextOnlyPackagingError(msg)
    if args.receipt.exists() and args.receipt.is_dir():
        msg = f"receipt path is a directory: {args.receipt}"
        raise TextOnlyPackagingError(msg)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.receipt.parent, prefix=".receipt-write-probe-"
    ) as receipt_probe:
        receipt_probe.write(b"write-probe")
    if args.output.exists() and any(args.output.iterdir()):
        msg = f"output directory is not empty: {args.output}"
        raise TextOnlyPackagingError(msg)
    raw_model_path = args.source / "model.safetensors"
    raw_model_sha256 = sha256_file(raw_model_path)
    if raw_model_sha256 != output_campaign["raw_model_safetensors_sha256"]:
        msg = "raw quantized model.safetensors hash does not match campaign"
        raise TextOnlyPackagingError(msg)
    source_config = json.loads((args.source / "config.json").read_text())
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    flattened = flatten_config(
        source_config, output_campaign["architecture"], output_campaign["text_model_type"]
    )
    tensors = load_file(str(args.source / "model.safetensors"), device="cpu")
    verify_source_index(source_index, set(tensors), checkpoint_prefix, artifact_layers_prefix)
    text_tensors = {}
    for source_key, tensor in tensors.items():
        target_key = tensor_key(source_key, checkpoint_prefix, artifact_layers_prefix)
        if target_key is not None:
            text_tensors[target_key] = tensor
    packed = sum(key.endswith("weight_packed") for key in text_tensors)
    scaled = sum(key.endswith("weight_scale") for key in text_tensors)
    expected_pairs = output_campaign["expected_packed_tensor_pairs"]
    if (packed, scaled) != (expected_pairs, expected_pairs):
        msg = f"expected {expected_pairs} packed/scaled tensor pairs, found {(packed, scaled)}"
        raise TextOnlyPackagingError(msg)
    tokenizer_files = verify_pinned_tokenizer(
        args.tokenizer_source, output_campaign["tokenizer_files"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output.name}.staging-", dir=args.output.parent
    ) as staging_directory:
        staging = Path(staging_directory)
        save_file(text_tensors, str(staging / "model.safetensors"), metadata={"format": "pt"})
        total_size = sum(tensor.numel() * tensor.element_size() for tensor in text_tensors.values())
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {key: "model.safetensors" for key in sorted(text_tensors)},
        }
        (staging / "config.json").write_text(json.dumps(flattened, indent=2, sort_keys=True) + "\n")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        for name in ("generation_config.json", "recipe.yaml"):
            source_file = args.source / name
            if source_file.is_file():
                shutil.copyfile(source_file, staging / name)
        copy_pinned_tokenizer(args.tokenizer_source, staging, tokenizer_files)
        files = {
            item.name: sha256_file(item) for item in sorted(staging.iterdir()) if item.is_file()
        }
        verify_staged_tokenizer_hashes(files, tokenizer_files)
        if args.output.exists():
            args.output.rmdir()
        staging.replace(args.output)
    receipt = {
        "schema": "mlx-gptq-calibration/mxfp4-text-only-packaging/2",
        "recorded_at": recorded_at(),
        "campaign": campaign["name"],
        "source": str(args.source),
        "source_model_sha256": raw_model_sha256,
        "output": str(args.output),
        "architectures": flattened["architectures"],
        "tensor_count": len(text_tensors),
        "packed_linear_tensors": packed,
        "scale_tensors": scaled,
        "tokenizer_source": str(args.tokenizer_source),
        "tokenizer_files": tokenizer_files,
        "output_files": files,
        "non_claims": [
            "This packaging pass changes config/key layout only; it does not alter packed values.",
            "Live generation and parity remain separate validation gates.",
        ],
    }
    if args.supersedes:
        receipt["supersedes"] = args.supersedes
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        required=True,
        help="verified source checkpoint supplying the pinned tokenizer byte-for-byte",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--supersedes",
        help="filename of the packaging receipt superseded by this run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_conversion(args)
    except (ImportError, KeyError, OSError, ValueError) as exc:
        print(f"text-only packaging failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
