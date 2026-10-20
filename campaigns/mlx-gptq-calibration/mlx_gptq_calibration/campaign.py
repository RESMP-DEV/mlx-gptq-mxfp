from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CampaignError(ValueError):
    pass


@dataclass(frozen=True)
class Campaign:
    path: Path
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def remote_root(self) -> str:
        return self.data["compute"].get(
            "remote_root", f".local/share/mlx-gptq-calibration/{self.name}"
        )

    @property
    def remote_model_dir(self) -> str:
        model = self.data["model"]
        repo_slug = model["id"].replace("/", "-")
        default = f"models/{repo_slug}-{model['revision'][:7]}"
        return self.data["compute"].get("remote_model_dir", default)

    def remote_path(self, *parts: str) -> str:
        suffix = "/".join(part.strip("/") for part in parts)
        return f"{self.remote_root.rstrip('/')}/{suffix}"


def load_campaign(path: str | Path) -> Campaign:
    campaign_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(campaign_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot load campaign {campaign_path}: {exc}") from exc

    required = {
        "schema_version",
        "name",
        "model",
        "dataset",
        "engine",
        "calibration",
        "compute",
        "output",
    }
    missing = required - data.keys()
    if missing:
        raise CampaignError(f"campaign is missing keys: {sorted(missing)}")
    if data["schema_version"] != 1:
        raise CampaignError(f"unsupported schema_version {data['schema_version']}")

    _require_sha(data["model"].get("revision"), "model.revision")
    _require_sha(data["dataset"].get("revision"), "dataset.revision")
    _require_sha(data["dataset"].get("sha256"), "dataset.sha256", length=64)
    _require_sha(data["engine"].get("commit"), "engine.commit")

    quant = data["calibration"]
    if quant["mode"] == "mxfp4" and (quant["bits"], quant["group_size"]) != (4, 32):
        raise CampaignError("mxfp4 requires bits=4 and group_size=32")
    if quant["nsamples"] <= 0 or quant["seqlen"] <= 0 or quant["batch_size"] <= 0:
        raise CampaignError("sample, sequence, and batch sizes must be positive")
    if not data["model"]["layers_attr"].endswith(".layers"):
        raise CampaignError("model.layers_attr must identify the decoder layer collection")

    packing = data.get("packing", {})
    for field in ("keep_bf16_regex",):
        if value := packing.get(field):
            _require_regex(value, f"packing.{field}")

    hybrid = data.get("hybrid")
    if hybrid is not None:
        source = hybrid.get("source", {})
        if not source.get("id"):
            raise CampaignError("hybrid.source.id is required")
        _require_sha(source.get("revision"), "hybrid.source.revision")
        _require_regex(hybrid.get("match"), "hybrid.match")
        for field in ("source_prefix", "target_prefix"):
            if not hybrid.get(field):
                raise CampaignError(f"hybrid.{field} is required")
    return Campaign(campaign_path, data)


def _require_sha(value: object, field: str, length: int = 40) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise CampaignError(f"{field} must be a {length}-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CampaignError(f"{field} must be hexadecimal") from exc


def _require_regex(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{field} must be a non-empty regular expression")
    try:
        re.compile(value)
    except re.error as exc:
        raise CampaignError(f"{field} is not a valid regular expression: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
