#!/usr/bin/env python3
"""Export DFlash2 weights directly onto the native MLX MXFP4 grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from mlx_gptq.artifacts import ArtifactWriter
from mlx_gptq.grid import rtn_quantize_mxfp
from safetensors.torch import load_file

_LAYER_SUFFIXES = frozenset(
    (
        "attention_conv.kernel_projection",
        "mlp_conv.kernel_projection",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )
)


def target_tensor_names(weights: dict[str, torch.Tensor]) -> list[str]:
    result = []
    for key, value in weights.items():
        if value.ndim != 2 or not key.endswith(".weight"):
            continue
        name = key[: -len(".weight")]
        if name in {"fc", "candidate_selector.hidden_projection"}:
            result.append(name)
            continue
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
            continue
        if parts[2] not in _LAYER_SUFFIXES:
            continue
        result.append(name)
    return sorted(result)


def export(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    source = Path(args.model)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    weights = load_file(str(source / "model.safetensors"), device="cpu")
    names = target_tensor_names(weights)
    if not names:
        raise RuntimeError("model contains no DFlash2 MXFP4 target tensors")

    writer = ArtifactWriter(
        str(output),
        {
            "model": str(source),
            "algorithm": "rtn-qat",
            "mode": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "storage_dtype": "bfloat16",
            "source": "qat" if args.qat_source else "bf16",
        },
    )
    tensors = {}
    metadata = {}
    for name in names:
        weight = weights[f"{name}.weight"].float()
        if weight.shape[-1] % 32:
            raise ValueError(f"{name} input width is not divisible by 32")
        q, scales, _biases, _dq = rtn_quantize_mxfp(weight[None], "mxfp4")
        tensors[name] = (q[0], scales[0], None)
        metadata[name] = {
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
            "kind": "dflash2-linear",
            "out_features": int(q.shape[1]),
            "in_features": int(q.shape[2]),
        }
    writer.write_layer(0, tensors, metadata)
    return {
        "schema_version": 1,
        "model": str(source),
        "output": str(output),
        "algorithm": "rtn-qat",
        "source": "qat" if args.qat_source else "bf16",
        "tensors": len(names),
        "elapsed_seconds": time.time() - started,
        "non_claims": [
            "RTN export does not perform Hessian-aware GPTQ error compensation.",
            "Tensor-layout success does not prove live acceptance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qat-source", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = export(args)
    payload = json.dumps(result, indent=2) + "\n"
    print(payload, end="")
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload)


if __name__ == "__main__":
    main()
