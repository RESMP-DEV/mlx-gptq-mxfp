#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class VllmValidationError(ValueError):
    pass


def recorded_at() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    try:
        return {
            "vllm": importlib.metadata.version("vllm"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        msg = f"required vLLM package is missing: {exc}"
        raise VllmValidationError(msg) from exc


def validate_quantization_config(
    config: Mapping[str, Any], expected_version: str | None = None
) -> dict[str, Any]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        msg = "model config has no quantization_config"
        raise VllmValidationError(msg)
    if quantization.get("quant_method") != "compressed-tensors":
        msg = f"wrong quant method: {quantization.get('quant_method')!r}"
        raise VllmValidationError(msg)
    if quantization.get("format") != "mxfp4-pack-quantized":
        msg = f"wrong compressed-tensors format: {quantization.get('format')!r}"
        raise VllmValidationError(msg)
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        msg = "quantization_config has no config groups"
        raise VllmValidationError(msg)
    if any(not isinstance(group, Mapping) for group in groups.values()):
        msg = "quantization config group entries must be objects"
        raise VllmValidationError(msg)
    weights = [group.get("weights") for group in groups.values()]
    if any(
        not isinstance(weight, Mapping)
        or weight.get("num_bits") != 4
        or weight.get("group_size") != 32
        or weight.get("type") != "float"
        for weight in weights
    ):
        msg = "quantization groups are not MXFP4 group-32 float weights"
        raise VllmValidationError(msg)
    if any(
        group.get("input_activations") is not None or group.get("output_activations") is not None
        for group in groups.values()
    ):
        msg = "MXFP4A16 must not quantize input or output activations"
        raise VllmValidationError(msg)
    if expected_version is not None and quantization.get("version") != expected_version:
        msg = f"compressed-tensors version mismatch: expected {expected_version}"
        raise VllmValidationError(msg)
    return dict(quantization)


def gpu_state() -> list[dict[str, Any]] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


def validate_gpu_state(
    gpus: list[dict[str, Any]] | None, minimum_compute_capability: int
) -> list[dict[str, Any]]:
    if not gpus:
        msg = "CUDA GPU state is unavailable; this validator must prove CUDA execution"
        raise VllmValidationError(msg)
    for gpu in gpus:
        capability = int(gpu["capability"][0])
        if capability < minimum_compute_capability:
            msg = (
                f"GPU compute capability {capability} is below campaign minimum "
                f"{minimum_compute_capability}"
            )
            raise VllmValidationError(msg)
    return gpus


def verify_model_directory(model: Path) -> None:
    required = ("config.json", "model.safetensors", "model.safetensors.index.json")
    if not model.is_dir():
        msg = f"model path is not a directory: {model}"
        raise VllmValidationError(msg)
    for name in required:
        if not (model / name).is_file():
            msg = f"model directory missing required file: {name}"
            raise VllmValidationError(msg)


def verify_artifact_identity(model: Path, campaign: Mapping[str, Any]) -> dict[str, str]:
    output = campaign["output"]
    expected_hashes = {
        "model.safetensors": output["text_model_safetensors_sha256"],
        "config.json": output["text_model_config_sha256"],
    }
    expected_hashes.update(output["tokenizer_files"])
    actual_hashes = {name: sha256_file(model / name) for name in expected_hashes}
    mismatches = {
        name: {"expected": expected, "actual": actual_hashes[name]}
        for name, expected in expected_hashes.items()
        if actual_hashes[name] != expected
    }
    if mismatches:
        msg = f"artifact hashes do not match campaign: {mismatches}"
        raise VllmValidationError(msg)
    return actual_hashes


def verify_receipt_path(model: Path, receipt: Path) -> None:
    model_resolved = model.resolve()
    receipt_resolved = receipt.resolve()
    if receipt_resolved == model_resolved or receipt_resolved.is_relative_to(model_resolved):
        msg = f"receipt path must not be inside the model directory: {receipt}"
        raise VllmValidationError(msg)


def run_validation(args: argparse.Namespace) -> None:
    campaign = json.loads(args.campaign.read_text())
    versions = package_versions()
    expected_vllm = args.expected_vllm or campaign["validation"]["vllm"]
    if versions["vllm"] != expected_vllm:
        msg = f"vLLM version mismatch: expected {expected_vllm}, got {versions['vllm']}"
        raise VllmValidationError(msg)
    verify_model_directory(args.model)
    verify_receipt_path(args.model, args.receipt)
    artifact_hashes = verify_artifact_identity(args.model, campaign)
    config = json.loads((args.model / "config.json").read_text())
    if not isinstance(config, Mapping):
        msg = "model config.json must contain an object"
        raise VllmValidationError(msg)
    quantization = validate_quantization_config(config, campaign["runtime"]["compressed_tensors"])
    gpus = validate_gpu_state(gpu_state(), campaign["validation"]["minimum_compute_capability"])
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        quantization="compressed-tensors",
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    started = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    elapsed = time.perf_counter() - started
    if len(outputs) != 1 or not outputs[0].outputs:
        msg = "vLLM returned no generation output"
        raise VllmValidationError(msg)
    completion = outputs[0].outputs[0]
    receipt = {
        "schema": "mlx-gptq-calibration/vllm-mxfp4-validation/1",
        "recorded_at": recorded_at(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runtime": versions,
        "gpus": gpus,
        "campaign": campaign["name"],
        "model": str(args.model),
        "artifact_hashes": artifact_hashes,
        "architectures": config.get("architectures"),
        "quantization_config": quantization,
        "settings": {
            "dtype": args.dtype,
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "prompt": args.prompt,
        "output": completion.text,
        "token_ids": list(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "generation_elapsed_seconds": round(elapsed, 3),
        "non_claims": [
            "One deterministic generation proves CUDA execution, not accuracy or throughput.",
            "Parity and performance require a separate matched BF16 control protocol.",
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--expected-vllm")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_validation(args)
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"vLLM validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
