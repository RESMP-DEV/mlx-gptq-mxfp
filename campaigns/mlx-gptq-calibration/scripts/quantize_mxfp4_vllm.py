#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class QuantizationContractError(ValueError):
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
            "llmcompressor": importlib.metadata.version("llmcompressor"),
            "compressed_tensors": importlib.metadata.version("compressed-tensors"),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "causal_conv1d": importlib.metadata.version("causal-conv1d"),
            "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        msg = f"required quantization package is missing: {exc}"
        raise QuantizationContractError(msg) from exc


def llmcompressor_record_sha256() -> str:
    distribution = importlib.metadata.distribution("llmcompressor")
    records = [file for file in distribution.files or [] if file.name == "RECORD"]
    if len(records) != 1:
        msg = f"cannot identify one llmcompressor installation RECORD: {records}"
        raise QuantizationContractError(msg)
    record_path = Path(str(distribution.locate_file(records[0])))
    if not record_path.is_file():
        msg = f"llmcompressor installation RECORD is missing: {record_path}"
        raise QuantizationContractError(msg)
    return sha256_file(record_path)


def verify_runtime(campaign: Mapping[str, Any]) -> dict[str, str]:
    actual = package_versions()
    expected = campaign["runtime"]
    expected_versions = {
        name: value for name, value in expected.items() if name != "llmcompressor_record_sha256"
    }
    mismatches = {
        name: {"expected": expected_versions[name], "actual": actual[name]}
        for name in expected_versions
        if actual[name] != expected_versions[name]
    }
    if mismatches:
        msg = f"runtime package versions do not match campaign: {mismatches}"
        raise QuantizationContractError(msg)
    engine = campaign.get("engine")
    if not isinstance(engine, Mapping):
        msg = "campaign is missing the engine section"
        raise QuantizationContractError(msg)
    commit_scope = engine.get("commit_scope")
    record_sha256 = llmcompressor_record_sha256()
    if commit_scope != "recipe_reference_not_installed_build_provenance":
        msg = "engine commit must explicitly be scoped as a recipe reference"
        raise QuantizationContractError(msg)
    if record_sha256 != expected["llmcompressor_record_sha256"]:
        msg = "llmcompressor installation RECORD hash does not match campaign"
        raise QuantizationContractError(msg)
    return {**actual, "llmcompressor_record_sha256": record_sha256}


def load_calibration(path: Path, expected_sha256: str, shape: tuple[int, int]) -> Any:
    import numpy as np

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        msg = f"calibration matrix hash mismatch: {actual_sha256} != {expected_sha256}"
        raise QuantizationContractError(msg)
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != shape:
        msg = f"calibration matrix shape mismatch: {matrix.shape} != {shape}"
        raise QuantizationContractError(msg)
    if np.isnan(matrix).any():
        msg = "calibration matrix contains NaN values"
        raise QuantizationContractError(msg)
    if matrix.min() < 0:
        msg = "calibration matrix contains negative token IDs"
        raise QuantizationContractError(msg)
    return matrix


def verify_source_receipt(
    receipt: Mapping[str, Any], campaign: Mapping[str, Any], model_path: Path
) -> dict[str, Any]:
    source_model = {
        "status": receipt.get("status"),
        "repo": receipt.get("repo"),
        "requested_revision": receipt.get("requested_revision"),
        "resolved_revision": receipt.get("resolved_revision"),
        "output": receipt.get("output"),
        "failures": receipt.get("failures"),
    }
    expected_revision = campaign["model"]["revision"]
    mismatches = {
        "status": source_model["status"] != "verified",
        "repo": source_model["repo"] != campaign["model"]["id"],
        "requested_revision": source_model["requested_revision"] != expected_revision,
        "resolved_revision": source_model["resolved_revision"] != expected_revision,
        "model_path": Path(str(source_model["output"])).resolve() != model_path.resolve(),
        "failures": bool(source_model["failures"]),
    }
    if any(mismatches.values()):
        msg = f"source model verification receipt does not match campaign: {source_model}"
        raise QuantizationContractError(msg)
    return source_model


def prepare_receipt_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        msg = f"receipt path is a directory: {path}"
        raise QuantizationContractError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".receipt-write-probe-") as probe:
        probe.write(b"write-probe")


def prepare_output_and_receipt_paths(output: Path, receipt: Path) -> None:
    output_resolved = output.resolve()
    receipt_resolved = receipt.resolve()
    if receipt_resolved == output_resolved or receipt_resolved.is_relative_to(output_resolved):
        msg = f"receipt path must not be inside the output directory: {receipt}"
        raise QuantizationContractError(msg)
    if output.exists() and not output.is_dir():
        msg = f"output path is not a directory: {output}"
        raise QuantizationContractError(msg)
    if output.exists() and any(output.iterdir()):
        msg = f"output directory is not empty: {output}"
        raise QuantizationContractError(msg)
    prepare_receipt_path(receipt)
    output.mkdir(parents=True, exist_ok=True)


def checkpoint_manifest_sha256(model_path: Path) -> str:
    files = sorted(path for path in model_path.rglob("*") if path.is_file())
    if not files:
        msg = f"source model directory has no regular files: {model_path}"
        raise QuantizationContractError(msg)
    entries = []
    for path in files:
        entries.append(
            f"{sha256_file(path)}\t{path.stat().st_size}\t{path.relative_to(model_path).as_posix()}"
        )
    return hashlib.sha256(("\n".join(entries) + "\n").encode()).hexdigest()


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return name
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        msg = f"invalid torch dtype in campaign: {name}"
        raise QuantizationContractError(msg)
    return value


def output_hashes(path: Path) -> dict[str, str]:
    return {item.name: sha256_file(item) for item in sorted(path.iterdir()) if item.is_file()}


def gpu_state() -> list[dict[str, Any]] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]


def validate_quantization_gpus(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    gpus = gpu_state()
    if not gpus:
        msg = "the cuda-mxfp4-vllm recipe requires a CUDA device"
        raise QuantizationContractError(msg)
    minimum_vram_bytes = campaign["compute"]["minimum_vram_mib"] * 1024 * 1024
    minimum_capability = int(campaign["validation"]["minimum_compute_capability"])
    for gpu in gpus:
        if gpu["total_bytes"] < minimum_vram_bytes:
            msg = f"CUDA device has insufficient VRAM: {gpu}"
            raise QuantizationContractError(msg)
        if gpu["capability"][0] < minimum_capability:
            msg = f"CUDA device is below minimum compute capability: {gpu}"
            raise QuantizationContractError(msg)
    return gpus


def _write_receipt(
    *,
    path: Path,
    campaign: Mapping[str, Any],
    model_path: Path,
    calibration_path: Path,
    output_path: Path,
    elapsed_seconds: float,
    runtime: dict[str, str],
) -> None:
    receipt = {
        "schema": "mlx-gptq-calibration/cuda-mxfp4-vllm/1",
        "recorded_at": recorded_at(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "campaign": campaign["name"],
        "source_model": {
            "id": campaign["model"]["id"],
            "revision": campaign["model"]["revision"],
            "path": str(model_path),
            "verification": campaign["model"].get("verification"),
        },
        "calibration": {
            "id": campaign["dataset"]["id"],
            "revision": campaign["dataset"]["revision"],
            "path": str(calibration_path),
            "sha256": campaign["dataset"]["sha256"],
            "shape": [campaign["calibration"]["nsamples"], campaign["calibration"]["seqlen"]],
        },
        "runtime": runtime,
        "gpus": gpu_state(),
        "model_class": campaign["model"].get("model_class"),
        "recipe": {
            "scheme": campaign["output"]["scheme"],
            "format": campaign["output"]["format"],
            "pipeline": campaign["compute"]["pipeline"],
            "group_size": campaign["calibration"]["group_size"],
            "dampening_frac": campaign["calibration"]["damp"],
            "actorder": campaign["calibration"]["actorder"],
        },
        "output": {
            "path": str(output_path),
            "files": output_hashes(output_path),
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "non_claims": [
            "This receipt proves tool execution and packed output creation, not model accuracy.",
            "A separate live vLLM generation and parity/performance run is required before "
            "serving.",
        ],
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run_quantization(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    campaign = copy.deepcopy(json.loads(args.campaign.read_text()))
    calibration = campaign["calibration"]
    if (calibration["bits"], calibration["group_size"]) != (4, 32):
        msg = "this script only implements the pinned MXFP4 bits=4, group_size=32 recipe"
        raise QuantizationContractError(msg)
    runtime = verify_runtime(campaign)
    quantization_gpus = validate_quantization_gpus(campaign)
    matrix = load_calibration(
        args.calibration,
        campaign["dataset"]["sha256"],
        (calibration["nsamples"], calibration["seqlen"]),
    )
    source_receipt = json.loads(args.source_receipt.read_text())
    source_model = verify_source_receipt(source_receipt, campaign, args.model)
    expected_manifest = campaign["model"].get("pipeline_content_manifest_sha256")
    actual_manifest = checkpoint_manifest_sha256(args.model)
    if actual_manifest != expected_manifest:
        msg = f"source model content manifest mismatch: {actual_manifest} != {expected_manifest}"
        raise QuantizationContractError(msg)
    source_model["pipeline_content_manifest_sha256"] = actual_manifest
    campaign["model"]["verification"] = source_model
    if args.verify_only:
        payload = {
            "schema": "mlx-gptq-calibration/cuda-mxfp4-verify-only/1",
            "recorded_at": recorded_at(),
            "campaign": campaign["name"],
            "runtime": runtime,
            "gpus": quantization_gpus,
            "calibration_shape": list(matrix.shape),
            "source_model": source_model,
            "non_claims": [
                "This verify-only run proves preflight identity and runtime pinning; "
                "it does not rerun quantization."
            ],
        }
        print(json.dumps(payload, sort_keys=True))
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    prepare_output_and_receipt_paths(args.output, args.receipt)
    dataset = Dataset.from_dict({"input_ids": matrix.tolist()})
    model_dtype = resolve_torch_dtype(torch, calibration["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    campaign["model"]["model_class"] = type(model).__name__
    torch.manual_seed(calibration["seed"])
    recipe = GPTQModifier(
        targets="Linear",
        scheme=campaign["output"]["scheme"],
        ignore=["lm_head"],
        dampening_frac=calibration["damp"],
        actorder=calibration["actorder"],
    )
    started = time.perf_counter()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        pipeline=campaign["compute"]["pipeline"],
        sequential_targets=[calibration["sequential_target"]],
        sequential_offload_device="cpu",
        num_calibration_samples=calibration["nsamples"],
        max_seq_length=calibration["seqlen"],
        batch_size=calibration["batch_size"],
        shuffle_calibration_samples=False,
    )
    model.save_pretrained(args.output, save_compressed=True)
    _write_receipt(
        path=args.receipt,
        campaign=campaign,
        model_path=args.model,
        calibration_path=args.calibration,
        output_path=args.output,
        elapsed_seconds=time.perf_counter() - started,
        runtime=runtime,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_quantization(args)
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"quantization failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
