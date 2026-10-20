from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from .campaign import CampaignError, sha256_file

MAX_HEADER_BYTES = 128 * 1024 * 1024


def _read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        size_raw = handle.read(8)
        if len(size_raw) != 8:
            raise CampaignError(f"invalid safetensors file {path}: missing header length")
        size = int.from_bytes(size_raw, "little")
        if size <= 0 or size > MAX_HEADER_BYTES:
            raise CampaignError(f"invalid safetensors file {path}: bad header length {size}")
        raw = handle.read(size)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid safetensors file {path}: malformed header") from exc
    return size, header


def _request_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    expected = None
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
        expected = end - start + 1
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        if expected is not None and status != 206:
            raise CampaignError(f"range request returned HTTP {status}: {url}")
        data = response.read(expected + 1 if expected is not None else None)
    if expected is not None:
        if len(data) != expected:
            raise CampaignError(
                f"range request returned {len(data)} bytes, expected {expected}: {url}"
            )
    return data


def _remote_header(repo_id: str, revision: str, shard: str) -> tuple[int, dict[str, Any]]:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{shard}"
    size = int.from_bytes(_request_bytes(url, 0, 7), "little")
    if size <= 0 or size > MAX_HEADER_BYTES:
        raise CampaignError(f"remote shard {shard} has invalid header length {size}")
    try:
        header = json.loads(_request_bytes(url, 8, 7 + size))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"remote shard {shard} has a malformed header") from exc
    return size, header


def _remote_json(repo_id: str, revision: str, filename: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    try:
        return json.loads(_request_bytes(url))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot decode {repo_id}@{revision}/{filename}") from exc


def _api_model(repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
    try:
        return json.loads(_request_bytes(url))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot decode Hub API response for {repo_id}@{revision}") from exc


def _padded_header(header: dict[str, Any]) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def fetch_overlay(
    repo_id: str,
    revision: str,
    match: str,
    output: Path,
) -> dict[str, Any]:
    """Fetch only matching tensors from a pinned Hub safetensors checkpoint."""
    live = _api_model(repo_id, revision)
    if live.get("sha") != revision:
        raise CampaignError(
            f"overlay source revision moved: pinned {revision}, live {live.get('sha')}"
        )
    index = _remote_json(repo_id, revision, "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    pattern = re.compile(match)
    selected = sorted(name for name in weight_map if pattern.search(name))
    if not selected:
        raise CampaignError(f"overlay regex matched no tensors: {match}")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for name in selected:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = _remote_header(repo_id, revision, shard)
        info = headers[shard][1].get(name)
        if not info:
            raise CampaignError(f"{name} is absent from the header of {shard}")
        entries.append((shard, name, info))
    entries.sort(key=lambda item: (item[0], item[2]["data_offsets"][0], item[1]))

    overlay_header: dict[str, Any] = {
        "__metadata__": {
            "source_repo": repo_id,
            "source_revision": revision,
            "match": match,
        }
    }
    offset = 0
    for _, name, info in entries:
        length = info["data_offsets"][1] - info["data_offsets"][0]
        overlay_header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length

    header_raw = _padded_header(overlay_header)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tensor_hashes: dict[str, str] = {}
    try:
        with tmp.open("wb") as handle:
            handle.write(len(header_raw).to_bytes(8, "little"))
            handle.write(header_raw)
            for shard, name, info in entries:
                source_header_size = headers[shard][0]
                start, end = info["data_offsets"]
                url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{shard}"
                raw = _request_bytes(
                    url,
                    8 + source_header_size + start,
                    8 + source_header_size + end - 1,
                )
                tensor_hashes[name] = hashlib.sha256(raw).hexdigest()
                handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)

    return {
        "source": {"id": repo_id, "revision": revision, "api_sha": live.get("sha")},
        "match": match,
        "output": str(output),
        "tensor_count": len(entries),
        "tensor_bytes": offset,
        "overlay_bytes": output.stat().st_size,
        "overlay_sha256": sha256_file(output),
        "tensor_sha256": tensor_hashes,
    }


def _clone_file(source: Path, destination: Path) -> None:
    clone = getattr(os, "clonefile", None)
    if clone is not None:
        clone(source, destination)
    else:
        shutil.copy2(source, destination)


def apply_overlay(
    model_path: Path,
    overlay_path: Path,
    source_prefix: str,
    target_prefix: str,
) -> dict[str, Any]:
    """Replace same-shape BF16 tensors in an MLX model without repacking."""
    model_path = model_path.expanduser().resolve()
    overlay_path = overlay_path.expanduser().resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise CampaignError(f"model has no safetensors index: {model_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    overlay_header_size, overlay_header = _read_header(overlay_path)

    target_headers: dict[str, tuple[int, dict[str, Any]]] = {}
    patches: dict[str, list[dict[str, Any]]] = {}
    for source_name, source_info in overlay_header.items():
        if source_name == "__metadata__":
            continue
        if not source_name.startswith(source_prefix):
            raise CampaignError(
                f"overlay tensor does not start with {source_prefix}: {source_name}"
            )
        target_name = target_prefix + source_name[len(source_prefix) :]
        shard = weight_map.get(target_name)
        if shard is None:
            raise CampaignError(f"packed model is missing overlay target {target_name}")
        if shard not in target_headers:
            target_headers[shard] = _read_header(model_path / shard)
        target_info = target_headers[shard][1].get(target_name)
        if target_info is None:
            raise CampaignError(f"packed shard {shard} is missing {target_name}")
        if source_info["dtype"] != "BF16" or target_info["dtype"] != "BF16":
            raise CampaignError(
                f"overlay requires BF16 source and target: {source_name} -> {target_name}"
            )
        if source_info["shape"] != target_info["shape"]:
            raise CampaignError(
                f"overlay shape mismatch: {source_name} {source_info['shape']} -> "
                f"{target_name} {target_info['shape']}"
            )
        source_length = source_info["data_offsets"][1] - source_info["data_offsets"][0]
        target_length = target_info["data_offsets"][1] - target_info["data_offsets"][0]
        if source_length != target_length:
            raise CampaignError(f"overlay byte-size mismatch for {target_name}")
        patches.setdefault(shard, []).append(
            {
                "source_name": source_name,
                "target_name": target_name,
                "source_offsets": source_info["data_offsets"],
                "target_offsets": target_info["data_offsets"],
            }
        )

    staged: list[tuple[Path, Path]] = []
    tensor_receipts: dict[str, dict[str, str]] = {}
    try:
        with overlay_path.open("rb") as overlay:
            for shard, shard_patches in patches.items():
                target = model_path / shard
                target_header_size = target_headers[shard][0]
                tmp = target.with_suffix(target.suffix + ".hybrid-tmp")
                tmp.unlink(missing_ok=True)
                _clone_file(target, tmp)
                staged.append((tmp, target))
                with target.open("rb") as before, tmp.open("r+b") as after:
                    for patch in shard_patches:
                        source_start, source_end = patch["source_offsets"]
                        target_start, target_end = patch["target_offsets"]
                        length = source_end - source_start
                        overlay.seek(8 + overlay_header_size + source_start)
                        raw = overlay.read(length)
                        if len(raw) != length:
                            raise CampaignError(f"overlay is truncated at {patch['source_name']}")
                        before.seek(8 + target_header_size + target_start)
                        old = before.read(length)
                        after.seek(8 + target_header_size + target_start)
                        after.write(raw)
                        tensor_receipts[patch["target_name"]] = {
                            "before_sha256": hashlib.sha256(old).hexdigest(),
                            "after_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    after.flush()
                    os.fsync(after.fileno())
        for tmp, target in staged:
            os.replace(tmp, target)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)

    metadata = overlay_header.get("__metadata__", {})
    return {
        "model": str(model_path),
        "overlay": str(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "source": {
            "id": metadata.get("source_repo"),
            "revision": metadata.get("source_revision"),
        },
        "source_prefix": source_prefix,
        "target_prefix": target_prefix,
        "tensor_count": len(tensor_receipts),
        "shards_modified": sorted(patches),
        "tensors": tensor_receipts,
    }
