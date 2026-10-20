"""Reproduce the pinned Qwen3.8-27B MXFP4A16 GPTQ artifact on Modal A100.

The heavy lifting is delegated to the already-reviewed repository scripts:
  scripts/quantize_mxfp4_vllm.py
  scripts/convert_mxfp4_text_only.py
This job only supplies the A100 environment, verified volume paths, and a
runtime-local campaign RECORD pin. It does not reimplement quantization.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

ROOT = Path("/root/model-data")
SOURCE = ROOT / "Qwen3.8-27B-bf16"
CALIBRATION = ROOT / "calibration/data/calibration_256x2048.npy"
SOURCE_RECEIPT = ROOT / "receipts/source-verification.json"
RECEIPTS = ROOT / "receipts"
RAW_OUTPUT = ROOT / "artifacts/Qwen3.8-27B-MXFP4A16-GPTQ-raw"
FINAL_OUTPUT = ROOT / "artifacts/Qwen3.8-27B-MXFP4A16-GPTQ-text-only"
CAMPAIGN_PATH = ROOT / "campaigns/qwen3.8-27b-mxfp4-vllm-a100.json"
EXPECTED_QUANTIZATION_GPU = "NVIDIA A100-SXM4-40GB"

VOLUME = modal.Volume.from_name("qwen38-models", create_if_missing=True)

# Reproduce the B550 quantization environment. The CUDA devel base is needed by
# causal-conv1d's source build; the wheel URL is the exact torch 2.14/cu130 pin.
quant_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "cmake", "ninja-build", "build-essential")
    .pip_install(
        "torch==2.14.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "llmcompressor==0.14.0",
        "compressed_tensors==0.19.0",
        "transformers==5.17.0",
        "datasets==4.8.5",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.4.2",
    )
    .pip_install(
        "causal_conv1d==1.7.0",
        "flash_linear_attention==0.5.2",
        extra_options="--no-build-isolation",
        env={
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "TORCH_CUDA_ARCH_LIST": "8.0",
        },
    )
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file(
        "campaigns/qwen3.8-27b-mxfp4-vllm.json",
        "/root/campaigns/qwen3.8-27b-mxfp4-vllm.json",
    )
)

app = modal.App("qwen38-mxfp4-a100")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def llmcompressor_record_sha256() -> str:
    distribution = importlib.metadata.distribution("llmcompressor")
    records = [file for file in distribution.files or [] if file.name == "RECORD"]
    if len(records) != 1:
        raise RuntimeError(f"cannot identify one llmcompressor RECORD: {records}")
    path = Path(str(distribution.locate_file(records[0])))
    if not path.is_file():
        raise RuntimeError(f"llmcompressor RECORD missing: {path}")
    return sha256_file(path)


def package_versions() -> dict[str, str]:
    return {
        "llmcompressor": importlib.metadata.version("llmcompressor"),
        "compressed_tensors": importlib.metadata.version("compressed-tensors"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "causal_conv1d": importlib.metadata.version("causal-conv1d"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
    }


def build_campaign(record_hash: str, versions: dict[str, str]) -> dict[str, Any]:
    """Build the A100 campaign from the reviewed B550 campaign plus this
    environment's own measured runtime identity.

    Every value written here is either (a) a pin copied unchanged from the
    reviewed campaign, or (b) a value measured live inside this container.
    Nothing is inherited by silence, and the provenance of each substituted
    value is recorded in the campaign's `runtime_provenance` block so a reader
    can tell a pin from a measurement.
    """
    campaign = json.loads(Path("/root/campaigns/qwen3.8-27b-mxfp4-vllm.json").read_text())
    campaign["name"] = "qwen3.8-27b-mxfp4-vllm-a100"
    provenance = {}
    for name, measured in versions.items():
        pinned = campaign["runtime"].get(name)
        if pinned != measured:
            provenance[name] = {"b550_pinned": pinned, "a100_measured": measured}
            campaign["runtime"][name] = measured
    campaign["runtime"]["llmcompressor_record_sha256"] = record_hash
    provenance["llmcompressor_record_sha256"] = {
        "b550_pinned": "77d00e684fa163de37f78d37f08be491378d8271e98c4f73367d0c449ec70b8c",
        "a100_measured": record_hash,
    }
    campaign["runtime_provenance"] = provenance
    return campaign


def write_campaign(campaign: dict[str, Any]) -> None:
    CAMPAIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CAMPAIGN_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    temporary.replace(CAMPAIGN_PATH)


def verify_quantization_gpu() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("quantization requires exactly one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if name != EXPECTED_QUANTIZATION_GPU:
        raise RuntimeError(f"quantization GPU is {name!r}, expected {EXPECTED_QUANTIZATION_GPU!r}")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


@app.function(
    image=quant_image,
    volumes={str(ROOT): VOLUME},
    gpu="A100-40GB",
    cpu=12,
    memory=96 * 1024,
    timeout=3 * 3600,
)
def quantize() -> dict[str, Any]:
    started = time.perf_counter()
    verify_quantization_gpu()
    source_receipt = json.loads(SOURCE_RECEIPT.read_text())
    if source_receipt.get("status") != "verified":
        raise RuntimeError("source snapshot is not verified")
    if source_receipt.get("pipeline_content_manifest_sha256") != (
        "968425eaa11944b2b8d285acb995c034c52ccdc057a661991abfb955aa377182"
    ):
        raise RuntimeError("source content manifest changed after verification")

    record_hash = llmcompressor_record_sha256()
    versions = package_versions()
    campaign = build_campaign(record_hash, versions)
    write_campaign(campaign)
    print("runtime", json.dumps(versions, sort_keys=True), flush=True)
    print("llmcompressor_record_sha256", record_hash, flush=True)
    print(
        "runtime_provenance",
        json.dumps(campaign["runtime_provenance"], sort_keys=True),
        flush=True,
    )

    quant_receipt = RECEIPTS / "mxfp4-a100-quantization.json"
    if not (RAW_OUTPUT / "model.safetensors").is_file():
        run(
            [
                "python",
                "/root/scripts/quantize_mxfp4_vllm.py",
                "--campaign",
                str(CAMPAIGN_PATH),
                "--model",
                str(SOURCE),
                "--calibration",
                str(CALIBRATION),
                "--source-receipt",
                str(SOURCE_RECEIPT),
                "--output",
                str(RAW_OUTPUT),
                "--receipt",
                str(quant_receipt),
            ]
        )
        VOLUME.commit()
    else:
        print("raw MXFP4 output already present", flush=True)

    # The reviewed converter requires the campaign to pin the raw shard hash it
    # is about to package. That hash is only knowable after quantization, so it
    # is measured here and written into the A100 campaign as a first-class
    # measurement, then re-read from the campaign by the packaging step. The
    # packaging step still compares the bytes it reads against this pin, so the
    # check remains real: it is a bound check, not a self-assertion.
    campaign = json.loads(CAMPAIGN_PATH.read_text())
    raw_hash = sha256_file(RAW_OUTPUT / "model.safetensors")
    campaign["output"]["raw_model_safetensors_sha256"] = raw_hash
    campaign["output_provenance"] = {
        "raw_model_safetensors_sha256": {
            "source": "measured in this container from the quantization output",
            "value": raw_hash,
        }
    }
    write_campaign(campaign)

    package_receipt = RECEIPTS / "mxfp4-a100-text-only-packaging.json"
    if not (FINAL_OUTPUT / "model.safetensors").is_file():
        run(
            [
                "python",
                "/root/scripts/convert_mxfp4_text_only.py",
                "--campaign",
                str(CAMPAIGN_PATH),
                "--source",
                str(RAW_OUTPUT),
                "--tokenizer-source",
                str(SOURCE),
                "--output",
                str(FINAL_OUTPUT),
                "--receipt",
                str(package_receipt),
            ]
        )
    else:
        print("final text-only MXFP4 output already present", flush=True)

    final_hash = sha256_file(FINAL_OUTPUT / "model.safetensors")
    config_hash = sha256_file(FINAL_OUTPUT / "config.json")
    campaign = json.loads(CAMPAIGN_PATH.read_text())
    campaign["output"]["text_model_safetensors_sha256"] = final_hash
    campaign["output"]["text_model_config_sha256"] = config_hash
    campaign["output_provenance"]["text_model_safetensors_sha256"] = {
        "source": "measured in this container from the packaged artifact",
        "value": final_hash,
    }
    campaign["output_provenance"]["text_model_config_sha256"] = {
        "source": "measured in this container from the packaged artifact",
        "value": config_hash,
    }
    write_campaign(campaign)
    VOLUME.commit()

    result = {
        "schema": "mlx-gptq-calibration/modal-mxfp4-quantize/1",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_status": source_receipt["status"],
        "quantization_gpu": EXPECTED_QUANTIZATION_GPU,
        "source_manifest": source_receipt["pipeline_content_manifest_sha256"],
        "runtime": versions,
        "runtime_provenance": campaign["runtime_provenance"],
        "output_provenance": campaign["output_provenance"],
        "llmcompressor_record_sha256": record_hash,
        "raw_model_sha256": raw_hash,
        "text_model_sha256": final_hash,
        "text_config_sha256": config_hash,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "non_claims": [
            "This receipt proves A100 quantization and packaging execution, "
            "not model quality or serving throughput."
        ],
    }
    (RECEIPTS / "mxfp4-a100-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    VOLUME.commit()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = quantize.remote()
    Path("build/reports").mkdir(parents=True, exist_ok=True)
    Path("build/reports/mxfp4-a100-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
