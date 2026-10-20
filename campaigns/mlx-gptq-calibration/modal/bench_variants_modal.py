"""Benchmark Qwen3.8-27B precision variants on one A100 and one held-out corpus."""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path("/root/model-data")
HOLDOUT = ROOT / "holdout/data/heldout/agentic-coding/agentic-coding-long-context.txt"
VOLUME = modal.Volume.from_name("qwen38-models", create_if_missing=True)

# The CUDA devel base is required: vLLM's torch.compile path needs nvcc, and
# a slim image without /usr/local/cuda fails during engine initialization.
# Match the validated B550 serving stack: vLLM 0.32.0 with torch 2.13/cu130,
# transformers 5.19, and the optimized Qwen GDN kernels. Without causal-conv1d
# and flash-linear-attention, Transformers silently falls back to reference
# PyTorch implementations and throughput is not representative.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("git", "cmake", "ninja-build", "build-essential")
    .pip_install(
        "torch==2.13.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "vllm==0.32.0",
        "transformers==5.19.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "numpy==2.3.4",
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
    .add_local_file("modal/bench_model.py", "/root/bench_model.py")
)
app = modal.App("qwen38-variant-bench")
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"


# The BF16 reference arm holds ~50 GiB of BF16 text weights, and A100 PCIe and
# SXM4 have different memory bandwidth. Modal exposes one generic A100-80GB
# request, so the child process rejects any live device other than the measured
# SXM4 name before loading model weights.
#
# The two arms are benchmarked in separate containers, and Modal can hand out
# different A100 SKUs (SXM4 vs PCIe) for the same selector. Those SKUs have
# different memory bandwidth, so a throughput comparison across them would be
# unfair. The bench therefore records and checks the live GPU identity in every
# receipt; the comparison is only valid when both arms report this exact name.
@app.function(
    image=image,
    volumes={str(ROOT): VOLUME},
    gpu="A100-80GB",
    cpu=8,
    memory=48 * 1024,
    timeout=3 * 3600,
)
def bench(model_path: str, label: str, receipt_path: str, mode: str) -> dict:
    """Run one measurement mode in its own process, so the GPU starts clean.

    Perplexity and throughput are separate remote calls. The BF16 arm is
    ~50 GiB of weights, so running a vLLM engine in the same process as a
    loaded Hugging Face model would exhaust the GPU.
    """
    import subprocess

    receipt = ROOT / receipt_path
    subprocess.run(
        [
            "python",
            "/root/bench_model.py",
            "--model",
            model_path,
            "--holdout",
            str(HOLDOUT),
            "--receipt",
            str(receipt),
            "--label",
            label,
            "--mode",
            mode,
            "--require-gpu",
            EXPECTED_GPU_NAME,
            "--seq-len",
            "2048",
            "--stride",
            "512",
            "--max-chunks",
            "64",
            "--max-tokens",
            "128",
            "--gpu-memory-utilization",
            "0.85",
            "--max-model-len",
            "4096",
        ],
        check=True,
    )
    VOLUME.commit()
    return json.loads(receipt.read_text())


@app.local_entrypoint()
def main() -> None:
    variants = {
        "bf16": (
            str(ROOT / "artifacts/Qwen3.8-27B-BF16-text-only"),
            "receipts/bench-bf16.json",
        ),
        "mxfp4-gptq": (
            str(ROOT / "artifacts/Qwen3.8-27B-MXFP4A16-GPTQ-text-only"),
            "receipts/bench-mxfp4-gptq.json",
        ),
    }
    import os

    only = os.environ.get("BENCH_ONLY")
    mode_only = os.environ.get("BENCH_MODE")
    results = {}
    for label, (model, receipt) in variants.items():
        if only and label != only:
            continue
        print(f"=== benchmarking {label}: {model} ===", flush=True)
        arm = {}
        for mode, suffix in (("perplexity", "perplexity"), ("throughput", "throughput")):
            if mode_only and mode != mode_only:
                continue
            mode_receipt = receipt.replace(".json", f"-{suffix}.json")
            print(f"--- {label} / {mode} ---", flush=True)
            part = bench.remote(model, label, mode_receipt, mode)
            arm[mode] = part
        first = next(iter(arm.values()))
        arm["identity"] = first["identity"]
        arm["gpus"] = first["gpus"]
        arm["runtime"] = first["runtime"]
        arm["holdout"] = first["holdout"]
        if "perplexity" in arm:
            arm["perplexity"] = arm["perplexity"]["perplexity"]
        if "throughput" in arm:
            arm["throughput"] = arm["throughput"]["throughput"]
        results[label] = arm
    Path("build/reports").mkdir(parents=True, exist_ok=True)
    Path("build/reports/variant-bench-summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
