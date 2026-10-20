"""Pure validation helpers for private Hugging Face model publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PublicationError(ValueError):
    pass


def require_basename(name: str, *, kind: str) -> str:
    if not name or "/" in name or "\\" in name or name in {".", ".."} or "\x00" in name:
        msg = f"invalid {kind} name: {name!r}"
        raise PublicationError(msg)
    return name


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"cannot read publication JSON object {path}: {exc}"
        raise PublicationError(msg) from exc
    if not isinstance(value, dict):
        msg = f"expected a JSON object: {path}"
        raise PublicationError(msg)
    return value


def verify_artifact_files(
    artifact: Path,
    packaging: dict[str, Any],
    *,
    model_shard_sha256: str,
    config_sha256: str,
) -> dict[str, str]:
    expected = packaging.get("output_files")
    if not isinstance(expected, dict) or not expected:
        msg = "packaging receipt has no non-empty output_files object"
        raise PublicationError(msg)
    for name in expected:
        require_basename(name, kind="artifact file")
    actual_names = {path.name for path in artifact.iterdir() if path.is_file()}
    if actual_names != set(expected):
        msg = (
            "artifact file set differs from packaging receipt: "
            f"missing={sorted(set(expected) - actual_names)} "
            f"unexpected={sorted(actual_names - set(expected))}"
        )
        raise PublicationError(msg)

    measured: dict[str, str] = {}
    for name in sorted(expected):
        measured[name] = _sha256_file(artifact / name)
        if measured[name] != expected[name]:
            msg = (
                f"artifact hash mismatch for {name}: "
                f"expected={expected[name]} measured={measured[name]}"
            )
            raise PublicationError(msg)
    if measured["model.safetensors"] != model_shard_sha256:
        msg = "measured model shard does not match the recommended artifact"
        raise PublicationError(msg)
    if measured["config.json"] != config_sha256:
        msg = "measured config does not match the recommended artifact"
        raise PublicationError(msg)
    return measured


def verify_result_matches_packaging(
    result: dict[str, Any],
    packaging: dict[str, Any],
) -> None:
    try:
        artifact = result["recommendation"]["artifact"]
        expected = {
            "model.safetensors": artifact["model_safetensors_sha256"],
            "config.json": artifact["config_sha256"],
        }
        output_files = packaging["output_files"]
    except (KeyError, TypeError) as exc:
        msg = "result and packaging receipts do not contain the required artifact fields"
        raise PublicationError(msg) from exc
    for name, digest in expected.items():
        if output_files.get(name) != digest:
            msg = f"result and packaging receipts disagree on {name}"
            raise PublicationError(msg)


def verify_receipt_equivalence(
    local_receipts: dict[str, Path],
    volume_receipts: dict[str, Path],
) -> None:
    if set(local_receipts) != set(volume_receipts):
        msg = (
            "local and volume receipt names differ: "
            f"local={sorted(local_receipts)} volume={sorted(volume_receipts)}"
        )
        raise PublicationError(msg)
    for name in sorted(local_receipts):
        if load_object(local_receipts[name]) != load_object(volume_receipts[name]):
            msg = f"local and volume receipts differ for {name}"
            raise PublicationError(msg)


def build_expected_files(
    artifact: Path,
    artifact_hashes: dict[str, str],
    model_card: Path,
    receipt_paths: dict[str, Path],
) -> dict[str, int]:
    files = {}
    for name in sorted(artifact_hashes):
        require_basename(name, kind="artifact file")
        files[name] = (artifact / name).stat().st_size
    files["README.md"] = model_card.stat().st_size
    for name in sorted(receipt_paths):
        require_basename(name, kind="receipt")
        path = receipt_paths[name]
        if not path.is_file():
            msg = f"publication receipt missing: {path}"
            raise PublicationError(msg)
        files[f"receipts/qwen38-precision-ladder/{name}"] = path.stat().st_size
    return files


def verify_remote_files(
    remote_entries: dict[str, Any],
    expected_files: dict[str, int],
) -> str:
    actual_files = {
        path: entry
        for path, entry in remote_entries.items()
        if getattr(entry, "size", None) is not None and path != ".gitattributes"
    }
    if set(actual_files) != set(expected_files):
        msg = (
            "remote file set differs from prepared publication: "
            f"missing={sorted(set(expected_files) - set(actual_files))} "
            f"unexpected={sorted(set(actual_files) - set(expected_files))}"
        )
        raise PublicationError(msg)
    for name, size in expected_files.items():
        entry = actual_files[name]
        if entry.size != size:
            msg = f"remote size mismatch for {name}: expected={size} actual={entry.size}"
            raise PublicationError(msg)
    model_entry = actual_files["model.safetensors"]
    lfs = getattr(model_entry, "lfs", None)
    if not lfs or getattr(lfs, "sha256", None) is None:
        msg = "remote model.safetensors does not expose LFS SHA-256 metadata"
        raise PublicationError(msg)
    return str(lfs.sha256)


def verify_remote_artifact_files(
    remote_entries: dict[str, Any],
    expected_files: dict[str, int],
) -> str:
    """Verify an existing remote model byte set without requiring current docs."""
    artifact_names = {
        name for name in expected_files if name != "README.md" and not name.startswith("receipts/")
    }
    if not artifact_names:
        msg = "prepared publication has no artifact files"
        raise PublicationError(msg)
    for name in sorted(artifact_names):
        entry = remote_entries.get(name)
        size = expected_files[name]
        if entry is None or getattr(entry, "size", None) != size:
            actual = getattr(entry, "size", None)
            msg = f"remote artifact size mismatch for {name}: expected={size} actual={actual}"
            raise PublicationError(msg)
    model_entry = remote_entries["model.safetensors"]
    lfs = getattr(model_entry, "lfs", None)
    if not lfs or getattr(lfs, "sha256", None) is None:
        msg = "remote model.safetensors does not expose LFS SHA-256 metadata"
        raise PublicationError(msg)
    return str(lfs.sha256)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
