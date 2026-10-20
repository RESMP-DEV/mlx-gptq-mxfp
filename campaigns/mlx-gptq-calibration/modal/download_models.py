"""Stage the pinned Qwen3.8-27B base, calibration matrix, and held-out corpus
onto the Modal `qwen38-models` volume. CPU-only, detached: no GPU billing.

Provenance (all pinned by revision):
  model      Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
  calib-mtx  RESMP-DEV/ptq-calibration-corpus @ ffa2e74 (data/.../calibration_256x2048.npy)
  holdout    RESMP-DEV/ptq-calibration-corpus @ 6ef2bbf (data/heldout/agentic-coding/...)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CALIB_ID = "RESMP-DEV/ptq-calibration-corpus"
CALIB_REV = "ffa2e74b9361d573167373140f10dfd30c69a327"
HOLDOUT_REV = "6ef2bbf949b94192ee8bb78d45a2fc840a10cff5"

VOL = "qwen38-models"
ROOT = Path("/root/model-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]==0.35.3")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

vol = modal.Volume.from_name(VOL, create_if_missing=True)

app = modal.App("qwen38-stage")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={str(ROOT): vol},
    secrets=[modal.Secret.from_name("hf-token")],
    cpu=4.0,
    memory=8192,
    timeout=6 * 3600,
)
def stage() -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    started = time.perf_counter()
    model_dir = ROOT / "Qwen3.8-27B-bf16"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=== downloading BF16 base ===", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REV,
        local_dir=str(model_dir),
        max_workers=8,
    )

    print("=== downloading calibration matrix ===", flush=True)
    calib_dir = ROOT / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=CALIB_ID,
        revision=CALIB_REV,
        filename="data/calibration_256x2048.npy",
        repo_type="dataset",
        local_dir=str(calib_dir),
    )

    print("=== downloading holdout corpus ===", flush=True)
    holdout_dir = ROOT / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "data/heldout/agentic-coding/agentic-coding-long-context.txt",
        "data/heldout/agentic-coding/agentic-coding-receipt.json",
    ):
        hf_hub_download(
            repo_id=CALIB_ID,
            revision=HOLDOUT_REV,
            filename=fname,
            repo_type="dataset",
            local_dir=str(holdout_dir),
        )

    vol.commit()

    calib = calib_dir / "data" / "calibration_256x2048.npy"
    holdout = (
        holdout_dir / "data" / "heldout" / "agentic-coding" / "agentic-coding-long-context.txt"
    )
    shards = sorted(model_dir.glob("*.safetensors"))
    receipt = {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "model_dir": str(model_dir),
        "model_revision": MODEL_REV,
        "model_safetensors_count": len(shards),
        "calibration_sha256": sha256_file(calib),
        "holdout_sha256": sha256_file(holdout),
        "holdout_bytes": holdout.stat().st_size,
    }
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


@app.local_entrypoint()
def main() -> None:
    out = stage.remote()
    Path("build/reports/modal-stage-receipt.json").parent.mkdir(parents=True, exist_ok=True)
    Path("build/reports/modal-stage-receipt.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
