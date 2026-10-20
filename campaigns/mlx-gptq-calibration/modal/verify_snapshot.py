"""Verify the staged Qwen3.8-27B snapshot on the Modal volume against the pins
published by HF for revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.

Every file's size is checked; every LFS-tracked file is sha256-verified against
its HF oid (HF LFS oids are sha256 of file content). Also emits the pipeline
content-manifest hash computed exactly like scripts/quantize_mxfp4_vllm.py's
checkpoint_manifest_sha256 so the A100 campaign can pin it, plus a
source-verification receipt the merged quantizer will accept.

The pins below are generated from the live HF API response, not transcribed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VOL = "qwen38-models"
ROOT = Path("/root/model-data")
MODEL_DIR = ROOT / "Qwen3.8-27B-bf16"
RECEIPTS = ROOT / "receipts"

PINNED_FILES = {
    ".gitattributes": {"size": 1570},
    "LICENSE": {"size": 11544},
    "README.md": {"size": 65012},
    "chat_template.jinja": {"size": 8952},
    "config.json": {"size": 4312},
    "crc32.txt": {"size": 238},
    "generation_config.json": {"size": 202},
    "merges.txt": {"size": 3353259},
    "model-00001-of-00018.safetensors": {
        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
        "size": 3966730552,
    },
    "model-00002-of-00018.safetensors": {
        "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
        "size": 3043080328,
    },
    "model-00003-of-00018.safetensors": {
        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
        "size": 2542796952,
    },
    "model-00004-of-00018.safetensors": {
        "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
        "size": 3988973152,
    },
    "model-00005-of-00018.safetensors": {
        "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
        "size": 2099339864,
    },
    "model-00006-of-00018.safetensors": {
        "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
        "size": 3979553696,
    },
    "model-00007-of-00018.safetensors": {
        "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
        "size": 2108759344,
    },
    "model-00008-of-00018.safetensors": {
        "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
        "size": 3979553696,
    },
    "model-00009-of-00018.safetensors": {
        "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
        "size": 2108759344,
    },
    "model-00010-of-00018.safetensors": {
        "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
        "size": 3979553696,
    },
    "model-00011-of-00018.safetensors": {
        "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
        "size": 2108759344,
    },
    "model-00012-of-00018.safetensors": {
        "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
        "size": 3979553696,
    },
    "model-00013-of-00018.safetensors": {
        "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
        "size": 2108759344,
    },
    "model-00014-of-00018.safetensors": {
        "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
        "size": 3979553696,
    },
    "model-00015-of-00018.safetensors": {
        "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
        "size": 2108759344,
    },
    "model-00016-of-00018.safetensors": {
        "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
        "size": 3979564040,
    },
    "model-00017-of-00018.safetensors": {
        "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
        "size": 2108759344,
    },
    "model-00018-of-00018.safetensors": {
        "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
        "size": 3392197344,
    },
    "model.safetensors.index.json": {"size": 112216},
    "preprocessor_config.json": {"size": 390},
    "tokenizer.json": {
        "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        "size": 12809320,
    },
    "tokenizer_config.json": {"size": 17928},
    "video_preprocessor_config.json": {"size": 385},
    "vocab.json": {"size": 6722759},
}

image = modal.Image.debian_slim(python_version="3.12")
vol = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App("qwen38-verify")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_manifest(model_dir: Path) -> str:
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    entries = [
        f"{sha256_file(p)}\t{p.stat().st_size}\t{p.relative_to(model_dir).as_posix()}"
        for p in files
    ]
    return hashlib.sha256((chr(10).join(entries) + chr(10)).encode()).hexdigest()


@app.function(image=image, volumes={str(ROOT): vol}, cpu=4.0, memory=8192, timeout=2 * 3600)
def verify() -> dict:
    started = time.perf_counter()
    # Remove HF local_dir metadata so the snapshot equals the canonical repo
    # file set and the content manifest is not polluted by cache blobs.
    cache = MODEL_DIR / ".cache"
    removed_cache = cache.exists()
    if removed_cache:
        shutil.rmtree(cache)

    failures: list[str] = []
    per_file = {}
    for rel, pin in sorted(PINNED_FILES.items()):
        path = MODEL_DIR / rel
        if not path.is_file():
            failures.append(f"missing:{rel}")
            per_file[rel] = {"status": "missing"}
            continue
        size = path.stat().st_size
        record = {"size": size, "expected_size": pin["size"]}
        if size != pin["size"]:
            failures.append(f"size:{rel}")
            record["status"] = "size_mismatch"
        expected_sha = pin.get("sha256")
        if expected_sha:
            actual = sha256_file(path)
            record["sha256"] = actual
            if actual != expected_sha:
                failures.append(f"sha256:{rel}")
                record["status"] = "sha256_mismatch"
            else:
                record["status"] = "verified_sha256"
        else:
            record["status"] = record.get("status", "size_verified_non_lfs")
        per_file[rel] = record

    unexpected = sorted(
        p.relative_to(MODEL_DIR).as_posix()
        for p in MODEL_DIR.rglob("*")
        if p.is_file() and p.relative_to(MODEL_DIR).as_posix() not in PINNED_FILES
    )
    if unexpected:
        failures.append(f"unexpected_files:{len(unexpected)}")

    manifest = content_manifest(MODEL_DIR)
    status = "verified" if not failures else "failed"
    verified = {
        "schema": "mlx-gptq-calibration/modal-source-verification/1",
        "repo": MODEL_ID,
        "repo_type": "model",
        "requested_revision": MODEL_REV,
        "resolved_revision": MODEL_REV,
        "output": str(MODEL_DIR),
        "status": status,
        "failures": failures,
        "file_count": len(PINNED_FILES),
        "verified_file_count": sum(
            1
            for r in per_file.values()
            if str(r.get("status", "")).startswith(("verified", "size_verified"))
        ),
        "total_bytes": sum(r.get("size", 0) for r in per_file.values()),
        "pipeline_content_manifest_sha256": manifest,
        "removed_hf_cache_dir": removed_cache,
        "unexpected_files": unexpected,
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "per_file": per_file,
    }
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / "source-verification.json").write_text(
        json.dumps(verified, indent=2, sort_keys=True) + "\n"
    )
    vol.commit()
    summary = {k: v for k, v in verified.items() if k != "per_file"}
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    out = verify.remote()
    Path("build/reports").mkdir(parents=True, exist_ok=True)
    Path("build/reports/modal-source-verification-summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(out, indent=2, sort_keys=True))
