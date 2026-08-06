"""Compose a standard MLX model from module-level quantization sources.

This is intentionally a weight-level operation: every selected module is copied
with its packed weight and quantization metadata intact.  It does not dequantize
or requantize either source.

Example::

    python -m mlx_gptq.mix \
      --base ./model-gptq-mxfp4 \
      --override ./model-gptq-mxfp8 \
      --match '\\.conv\\.(in_proj|out_proj)$' \
      --output ./model-gptq-mixed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def _load_weights(path: Path):
    import mlx.core as mx

    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model safetensors in {path}")
    weights = {}
    for file in files:
        for key, value in mx.load(str(file)).items():
            if key in weights:
                raise RuntimeError(f"duplicate tensor {key} in {path}")
            weights[key] = value
    return weights


def _global_quantization(config: dict) -> dict | None:
    quant = config.get("quantization") or config.get("quantization_config")
    if not quant or not all(k in quant for k in ("group_size", "bits")):
        return None
    return {
        "group_size": quant["group_size"],
        "bits": quant["bits"],
        "mode": quant.get("mode", "affine"),
    }


def _module_paths(weights: dict) -> set[str]:
    paths = set()
    for key in weights:
        if not key.endswith(".weight"):
            continue
        path = key[: -len(".weight")]
        if f"{path}.scales" in weights:
            paths.add(path)
    return paths


def compose(base: str, override: str, output: str, match: str) -> dict:
    import mlx.core as mx

    base_path = Path(base).resolve()
    override_path = Path(override).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    with (base_path / "config.json").open() as file:
        base_config = json.load(file)
    with (override_path / "config.json").open() as file:
        override_config = json.load(file)

    base_weights = _load_weights(base_path)
    override_weights = _load_weights(override_path)
    pattern = re.compile(match)

    candidates = sorted(
        path
        for path in {
            *(key[: -len(".weight")] for key in base_weights if key.endswith(".weight")),
            *(key[: -len(".weight")] for key in override_weights if key.endswith(".weight")),
        }
        if pattern.search(path)
    )
    if not candidates:
        raise RuntimeError(f"pattern matched no module paths: {match}")

    override_quant = _global_quantization(override_config)
    base_quant = _global_quantization(base_config)
    if base_quant is None:
        raise RuntimeError("base model has no global MLX quantization config")

    mixed = dict(base_weights)
    replaced = []
    for path in candidates:
        weight_key = f"{path}.weight"
        if weight_key not in override_weights:
            raise RuntimeError(f"override is missing {weight_key}")

        # A module has one weight plus optional scale/bias quantization state.
        for suffix in ("weight", "scales", "biases"):
            mixed.pop(f"{path}.{suffix}", None)
        for suffix in ("weight", "scales", "biases"):
            key = f"{path}.{suffix}"
            if key in override_weights:
                mixed[key] = override_weights[key]
        replaced.append(path)

    quant_config = dict(base_config["quantization"])
    for path in replaced:
        quant_config[path] = override_quant if f"{path}.scales" in mixed else False
    base_config["quantization"] = quant_config
    base_config["quantization_config"] = quant_config

    output_path.mkdir(parents=True)
    for source in base_path.iterdir():
        if source.name == "config.json" or source.name.startswith("model") and source.suffix == ".safetensors":
            continue
        if source.name == "model.safetensors.index.json":
            continue
        if source.is_file():
            shutil.copy2(source, output_path / source.name)

    weight_file = output_path / "model.safetensors"
    mx.save_safetensors(str(weight_file), mixed, metadata={"format": "mlx"})
    with (output_path / "config.json").open("w") as file:
        json.dump(base_config, file, indent=2, sort_keys=True)
        file.write("\n")

    total_size = sum(value.nbytes for value in mixed.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: weight_file.name for key in sorted(mixed)},
    }
    with (output_path / "model.safetensors.index.json").open("w") as file:
        json.dump(index, file, indent=2, sort_keys=True)
        file.write("\n")

    digest = hashlib.sha256()
    with weight_file.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)

    return {
        "base": str(base_path),
        "override": str(override_path),
        "output": str(output_path),
        "match": match,
        "replaced_modules": len(replaced),
        "base_quantization": base_quant,
        "override_quantization": override_quant or "unquantized",
        "quantized_modules": len(_module_paths(mixed)),
        "weight_bytes": weight_file.stat().st_size,
        "weight_sha256": digest.hexdigest(),
        "module_paths": replaced,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--override", required=True)
    parser.add_argument("--match", required=True, help="regex matched against MLX module paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", help="optional JSON receipt path")
    args = parser.parse_args(argv)

    receipt = compose(args.base, args.override, args.output, args.match)
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    print(payload)
    if args.receipt:
        receipt_path = Path(args.receipt)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(payload + "\n")


if __name__ == "__main__":
    main()
