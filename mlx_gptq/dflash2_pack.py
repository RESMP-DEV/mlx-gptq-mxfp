"""Pack calibrated DFlash2 tensors into a native MLX MXFP4 draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .artifacts import ArtifactReader
from .packing import pack_mxfp4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.model)
    calibration = Path(args.calibration)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    reader = ArtifactReader(str(calibration))
    weights = load_file(str(source / "model.safetensors"))
    injected = 0
    for name in sorted(reader.tensors):
        info = reader.tensors[name]
        if info.get("mode") != "mxfp4":
            raise ValueError(f"{name} is not MXFP4")
        q, scales, _biases, _info = reader.get(name)
        if scales.dtype != torch.uint8:
            raise ValueError(f"{name} scales are not E8M0 uint8")
        original_key = f"{name}.weight"
        if original_key not in weights:
            raise KeyError(f"checkpoint has no {original_key}")
        weights[original_key] = torch.from_numpy(pack_mxfp4(q.numpy()))
        weights[f"{name}.scales"] = scales.contiguous()
        injected += 1

    model_path = output / "model.safetensors"
    save_file(weights, str(model_path))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization"] = {
        "group_size": int(reader.manifest["group_size"]),
        "bits": int(reader.manifest["bits"]),
        "mode": str(reader.manifest["mode"]),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copyfile(candidate, output / name)

    receipt = {
        "schema_version": 1,
        "source": str(source),
        "calibration": str(calibration),
        "mode": config["quantization"]["mode"],
        "bits": config["quantization"]["bits"],
        "group_size": config["quantization"]["group_size"],
        "calibrated_tensors": injected,
        "checkpoint_tensors": len(weights),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "selector_codebooks": "bf16",
        "non_claims": [
            "Packing proves tensor injection and native layout, not acceptance quality.",
            "The two selector codebooks remain BF16 in this first artifact.",
        ],
    }
    (output / "GPTQ_MXFP4_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = pack(args)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
