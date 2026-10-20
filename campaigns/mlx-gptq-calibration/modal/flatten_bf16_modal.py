"""Flatten the verified vision-language BF16 base to a text-only vLLM artifact."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path("/root/model-data")
SOURCE = ROOT / "Qwen3.8-27B-bf16"
OUTPUT = ROOT / "artifacts/Qwen3.8-27B-BF16-text-only"
RECEIPT = ROOT / "receipts/bf16-text-only-packaging.json"
VOLUME = modal.Volume.from_name("qwen38-models", create_if_missing=True)

# torch is required by safetensors.torch for the tensor surgery. This job
# only reshards tensors, so the CPU wheel is sufficient and avoids a GPU
# dependency for a non-GPU stage.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0+cu130", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("safetensors==0.8.0", "numpy==2.4.2")
    .add_local_file("modal/flatten_bf16_text_only.py", "/root/flatten_bf16_text_only.py")
)
app = modal.App("qwen38-flatten-bf16")


@app.function(
    image=image,
    volumes={str(ROOT): VOLUME},
    cpu=8,
    memory=80 * 1024,
    timeout=3 * 3600,
)
def flatten() -> dict:
    if (OUTPUT / "model.safetensors.index.json").is_file() and RECEIPT.is_file():
        return {"status": "already_present", "output": str(OUTPUT)}
    subprocess.run(
        [
            "python",
            "/root/flatten_bf16_text_only.py",
            "--source",
            str(SOURCE),
            "--output",
            str(OUTPUT),
            "--receipt",
            str(RECEIPT),
            "--shard-gib",
            "4.5",
        ],
        check=True,
    )
    VOLUME.commit()
    receipt = json.loads(RECEIPT.read_text())
    return {
        "status": "flattened",
        "tensor_count": receipt["tensor_count"],
        "shard_count": receipt["shard_count"],
        "output": str(OUTPUT),
    }


@app.local_entrypoint()
def main() -> None:
    result = flatten.remote()
    Path("build/reports").mkdir(parents=True, exist_ok=True)
    Path("build/reports/bf16-flatten-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
