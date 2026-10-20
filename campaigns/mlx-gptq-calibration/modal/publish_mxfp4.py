"""Publish the measured A100 Qwen3.8 MXFP4 artifact privately to Hugging Face."""

from __future__ import annotations

import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
from mlx_gptq_calibration.publication import (
    PublicationError,
    build_expected_files,
    load_object,
    verify_artifact_files,
    verify_receipt_equivalence,
    verify_remote_artifact_files,
    verify_remote_files,
    verify_result_matches_packaging,
)

ROOT = Path("/root/model-data")
ARTIFACT = ROOT / "artifacts/Qwen3.8-27B-MXFP4A16-GPTQ-text-only"
RECEIPTS = ROOT / "receipts"
LOCAL_RECEIPTS = Path("/root/local-receipts/qwen38-precision-ladder")
REPO_ID = "RESMP-DEV/Qwen3.8-27B-MXFP4A16-GPTQ-A100"
TAG = "a100-vllm-0.32.0-v3"
MODEL_SHARD = "1b6cf393189f59763dbd330cf85fefd2508cfca63db7c13aaea8d53da71e1dd7"
MODEL_CONFIG = "873538762602808f3b895701b479f2f085d917e390d85a9788aa26b796660055"

VOLUME = modal.Volume.from_name("qwen38-models", create_if_missing=True)
HF_TOKEN = modal.Secret.from_name("hf-token")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]==2.2.0")
    .add_local_file(
        "mlx_gptq_calibration/publication.py",
        "/root/mlx_gptq_calibration/publication.py",
    )
    .add_local_file(
        "modal/qwen38_mxfp4_model_card.md",
        "/root/qwen38_mxfp4_model_card.md",
    )
    .add_local_dir(
        "receipts/qwen38-precision-ladder",
        "/root/local-receipts/qwen38-precision-ladder",
    )
)
app = modal.App("qwen38-mxfp4-publish")

PUBLICATION_RECEIPTS = (
    "result.json",
    "source-verification.json",
    "bf16-text-only-packaging.json",
    "mxfp4-a100-quantization.json",
    "mxfp4-a100-text-only-packaging.json",
    "perplexity-bf16.json",
    "perplexity-mxfp4-gptq.json",
    "throughput-bf16.json",
    "throughput-mxfp4-gptq.json",
)

VOLUME_RECEIPTS = {
    "source-verification.json": "source-verification.json",
    "bf16-text-only-packaging.json": "bf16-text-only-packaging.json",
    "mxfp4-a100-quantization.json": "mxfp4-a100-quantization.json",
    "mxfp4-a100-text-only-packaging.json": "mxfp4-a100-text-only-packaging.json",
    "perplexity-bf16.json": "bench-bf16-perplexity.json",
    "perplexity-mxfp4-gptq.json": "bench-mxfp4-gptq-perplexity.json",
    "throughput-bf16.json": "bench-bf16-throughput.json",
    "throughput-mxfp4-gptq.json": "bench-mxfp4-gptq-throughput.json",
}

PUBLICATION_RECEIPT_NAMES = set(PUBLICATION_RECEIPTS) & set(VOLUME_RECEIPTS)
REFRESH_RECEIPTS = (
    "result.json",
    "perplexity-bf16.json",
    "perplexity-mxfp4-gptq.json",
    "throughput-bf16.json",
    "throughput-mxfp4-gptq.json",
)


def verify_artifact() -> dict[str, str]:
    """Verify every packaged artifact byte against its durable receipt."""
    packaging = load_object(RECEIPTS / "mxfp4-a100-text-only-packaging.json")
    return verify_artifact_files(
        ARTIFACT,
        packaging,
        model_shard_sha256=MODEL_SHARD,
        config_sha256=MODEL_CONFIG,
    )


def verify_ladder_result() -> None:
    """Cross-check the consolidated result against the artifact receipt."""
    result = load_object(LOCAL_RECEIPTS / "result.json")
    packaging = load_object(RECEIPTS / "mxfp4-a100-text-only-packaging.json")
    verify_result_matches_packaging(result, packaging)


def verify_local_receipts_match_volume() -> None:
    """Ensure durable publication receipts agree with their volume originals."""
    verify_receipt_equivalence(
        {
            local_name: LOCAL_RECEIPTS / local_name
            for local_name in sorted(PUBLICATION_RECEIPT_NAMES)
        },
        {
            local_name: RECEIPTS / volume_name
            for local_name, volume_name in VOLUME_RECEIPTS.items()
            if local_name in PUBLICATION_RECEIPT_NAMES
        },
    )


def publication_expected_files(artifact_hashes: dict[str, str]) -> dict[str, int]:
    """Build the exact direct-upload file set without duplicating the 17 GiB shard."""
    return build_expected_files(
        ARTIFACT,
        artifact_hashes,
        Path("/root/qwen38_mxfp4_model_card.md"),
        {name: LOCAL_RECEIPTS / name for name in PUBLICATION_RECEIPTS},
    )


def ensure_private_repo(api: Any) -> None:
    """Create the target privately and refuse an existing public repository."""
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(REPO_ID, expand=["private"])
    if not info.private:
        raise RuntimeError(f"refusing to publish because repository is public: {REPO_ID}")


def ensure_tag(api: Any, revision: str) -> None:
    """Keep the tag immutable once it identifies a publication revision."""
    refs = api.list_repo_refs(REPO_ID, repo_type="model")
    existing = next((ref for ref in refs.tags if ref.ref == f"refs/tags/{TAG}"), None)
    if existing is None:
        api.create_tag(REPO_ID, tag=TAG, revision=revision, repo_type="model")
        return
    if existing.target_commit != revision:
        raise RuntimeError(
            f"publication tag {TAG} already points at {existing.target_commit}, "
            f"not the uploaded revision {revision}"
        )


def commit_revision(commit: Any) -> str:
    revision = getattr(commit, "oid", None) or getattr(commit, "sha", None)
    if not revision:
        raise PublicationError("Hugging Face upload returned no revision")
    return str(revision)


@app.function(
    image=image,
    volumes={str(ROOT): VOLUME},
    secrets=[HF_TOKEN],
    cpu=2,
    memory=8 * 1024,
    timeout=4 * 3600,
)
def publish(refresh_only: bool = False) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    artifact_hashes = verify_artifact()
    verify_ladder_result()
    verify_local_receipts_match_volume()
    expected_files = publication_expected_files(artifact_hashes)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is absent from the Modal hf-token secret")
    api = HfApi(token=token)
    identity = api.whoami()
    ensure_private_repo(api)

    operations = []
    if refresh_only:
        current_tree = {
            entry.path: entry
            for entry in api.list_repo_tree(
                REPO_ID,
                repo_type="model",
                recursive=True,
                expand=True,
            )
        }
        existing_model_sha256 = verify_remote_artifact_files(current_tree, expected_files)
        if existing_model_sha256 != MODEL_SHARD:
            raise PublicationError("existing remote model shard does not match the pin")
        receipt_names = REFRESH_RECEIPTS
        commit_message = "Correct measured A100 evidence"
    else:
        operations = [
            CommitOperationAdd(name, str(ARTIFACT / name)) for name in sorted(artifact_hashes)
        ]
        receipt_names = PUBLICATION_RECEIPTS
        commit_message = "Publish measured A100 MXFP4A16 GPTQ artifact"
    operations.append(CommitOperationAdd("README.md", "/root/qwen38_mxfp4_model_card.md"))
    operations.extend(
        CommitOperationAdd(
            f"receipts/qwen38-precision-ladder/{name}",
            str(LOCAL_RECEIPTS / name),
        )
        for name in receipt_names
    )
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
        num_threads=5,
    )
    revision = commit_revision(commit)

    remote_tree = {
        entry.path: entry
        for entry in api.list_repo_tree(
            REPO_ID,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        )
    }
    remote_model_sha256 = verify_remote_files(remote_tree, expected_files)
    model_entry = remote_tree["model.safetensors"]
    if remote_model_sha256 != MODEL_SHARD:
        raise PublicationError("remote model.safetensors SHA-256 does not match the pin")
    ensure_tag(api, revision)

    result = {
        "schema": "mlx-gptq-calibration/hf-publication/2",
        "mode": "evidence-refresh" if refresh_only else "initial-publication",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repository": REPO_ID,
        "revision": revision,
        "tag": TAG,
        "private": True,
        "published_by": identity.get("name"),
        "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
        "artifact_sha256": artifact_hashes,
        "remote_file_count": len(expected_files),
        "remote_model_lfs_sha256": remote_model_sha256,
        "remote_model_bytes": model_entry.size,
        "commit_url": commit.commit_url,
        "receipts": [f"receipts/qwen38-precision-ladder/{name}" for name in PUBLICATION_RECEIPTS],
        "non_claims": [
            "Publication proves byte identity and remote metadata, not a new model-quality run.",
            "The measured recommendation remains specific to one A100-SXM4-80GB and vLLM 0.32.0.",
            "Task accuracy and batched serving capacity remain unmeasured.",
            "An evidence refresh does not re-quantize or alter model bytes.",
        ],
    }
    return result


@app.local_entrypoint()
def main() -> None:
    result = publish.remote(refresh_only=os.environ.get("PUBLICATION_REFRESH_ONLY") == "1")
    output = Path("receipts/qwen38-precision-ladder/hf-publication.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
