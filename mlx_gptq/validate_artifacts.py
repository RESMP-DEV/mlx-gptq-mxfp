"""Validate calibrated artifacts against their source checkpoint."""

from __future__ import annotations

import argparse
import json

import torch

from .artifacts import ArtifactReader
from .grid import MX_GROUP_SIZE, dequantize_artifact
from .sequential import RawShardIndex


def _group_max(value: torch.Tensor) -> torch.Tensor:
    rows, columns = value.shape
    grouped = value.reshape(rows, columns // MX_GROUP_SIZE, MX_GROUP_SIZE)
    return grouped.abs().amax(dim=-1)


def validate(model_path: str, calib_path: str) -> dict:
    reader = ArtifactReader(calib_path)
    shards = RawShardIndex(model_path)
    failures: list[dict[str, str]] = []
    worst: list[dict[str, float | str]] = []

    for name, info in reader.tensors.items():
        if not shards.has(name):
            failures.append({"name": name, "error": "missing source tensor"})
            continue
        source = shards.get(name).float()
        q, scales, biases, artifact_info = reader.get(name)
        mode = artifact_info.get("mode", info.get("mode", "affine"))
        group_size = int(artifact_info.get("group_size", info["group_size"]))
        if source.shape != q.shape:
            failures.append(
                {
                    "name": name,
                    "error": f"shape {tuple(source.shape)} != {tuple(q.shape)}",
                }
            )
            continue
        dequantized = dequantize_artifact(q, scales, biases, mode, group_size)
        if not bool(torch.isfinite(dequantized).all()):
            failures.append({"name": name, "error": "non-finite dequantized values"})
            continue

        source_max = _group_max(source)
        artifact_max = _group_max(dequantized)
        limit = torch.maximum(
            source_max.to(torch.float64) * 32.0,
            torch.full_like(source_max, 2.0**-126),
        )
        ratios = (artifact_max.to(torch.float64) / limit).max()
        if bool((artifact_max.to(torch.float64) > limit).any()):
            failures.append(
                {"name": name, "error": f"group range ratio {float(ratios):.6g}"}
            )
        worst.append(
            {
                "name": name,
                "ratio": round(float(ratios), 6),
                "source_max": float(source.abs().max()),
                "artifact_max": float(dequantized.abs().max()),
            }
        )

    worst.sort(key=lambda item: float(item["ratio"]), reverse=True)
    return {
        "model": model_path,
        "calibration": calib_path,
        "tensors": len(reader.tensors),
        "failures": failures,
        "worst": worst[:20],
        "status": "verified" if not failures else "failed",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = validate(args.model, args.calib)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
