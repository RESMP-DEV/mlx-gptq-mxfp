"""Sequential GPTQ calibration for a DFlash2 draft checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .artifacts import ArtifactWriter
from .capture import CaptureManager
from .gptq import solve_gptq
from .grid import mode_defaults

log = logging.getLogger(__name__)


def _load_reference(path: str, package: str, device: str):
    sys.path.insert(0, package)
    from dflash.model import DFlash2DraftModel

    model = DFlash2DraftModel.from_pretrained(path, dtype=torch.bfloat16)
    model.eval().to(device)
    return model


def _cycle_paths(capture_dir: Path) -> list[Path]:
    root = capture_dir / "cycles"
    names = sorted(path.name for path in root.iterdir())
    paths = [root / name / "tensors.safetensors" for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing capture tensors: {missing[:3]}")
    return paths


def _solve_weight(
    name: str,
    linear: torch.nn.Linear,
    rows: list[torch.Tensor],
    *,
    device: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not rows:
        raise RuntimeError(f"no captured rows for {name}")
    in_features = linear.in_features
    solve_device = (
        args.cpu_device
        if linear.in_features >= args.cpu_solve_min_in_features
        else device
    )
    hessian = torch.zeros(
        in_features, in_features, dtype=torch.float32, device=solve_device
    )
    token_count = 0
    for row in rows:
        for start in range(0, row.shape[0], 65536):
            x = row[start : start + 65536].to(solve_device, torch.float32)
            hessian.addmm_(x.T, x)
            token_count += x.shape[0]
    weight = linear.weight.detach().to(solve_device, torch.float32).unsqueeze(0)
    q, scales, _biases, weight_dq, rel = solve_gptq(
        weight,
        hessian.unsqueeze(0),
        args.bits,
        args.group_size,
        damp=args.damp,
        clip=args.clip,
        storage_dtype=torch.bfloat16,
        mode=args.mode,
        mxfp_algorithm=args.mxfp_algorithm,
    )
    log.info(
        "%s: tokens=%d rel_err=%.6f shape=%s",
        name,
        token_count,
        float(rel[0]),
        tuple(weight.shape[-2:]),
    )
    return q[0], scales[0], weight_dq[0], float(rel[0])


def _apply_dq(linear: torch.nn.Linear, weight_dq: torch.Tensor, device: str) -> None:
    linear.weight.data.copy_(weight_dq.to(linear.weight.dtype).to(device))


def _save_result(
    artifacts: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    writer: ArtifactWriter,
    layer_index: int,
) -> None:
    tensors = {
        name: (q, scales, None) for name, (q, scales, _dq, _rel) in artifacts.items()
    }
    meta = {
        name: {
            "bits": writer.manifest["bits"],
            "group_size": writer.manifest["group_size"],
            "mode": writer.manifest["mode"],
            "kind": "dflash2-linear",
            "out_features": int(q.shape[0]),
            "in_features": int(q.shape[1]),
            "rel_err": round(rel, 6),
        }
        for name, (q, scales, _dq, rel) in artifacts.items()
    }
    writer.write_layer(layer_index, tensors, meta)


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = args.device
    capture_dir = Path(args.capture_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = _load_reference(args.model, args.dflash_package, device)
    paths = _cycle_paths(capture_dir)
    if args.max_cycles:
        paths = paths[: args.max_cycles]
    if not paths:
        raise RuntimeError("capture has no cycles")

    payloads = [load_file(str(path)) for path in paths]
    for payload in payloads:
        required = ("context", "noise_embedding")
        missing = [name for name in required if name not in payload]
        if missing:
            raise KeyError(f"capture cycle missing tensors: {missing}")

    writer = ArtifactWriter(
        str(output),
        {
            "model": args.model,
            "capture_dir": str(capture_dir),
            "algorithm": args.mxfp_algorithm,
            "mode": args.mode,
            "bits": args.bits,
            "group_size": args.group_size,
            "storage_dtype": "bfloat16",
            "clip": args.clip,
            "damp": args.damp,
            "cycles": len(payloads),
        },
    )

    fc_rows: list[torch.Tensor] = []
    for payload in payloads:
        raw = payload.get("raw_context")
        projected = payload["context"]
        if raw is not None and raw.shape[1] == projected.shape[1]:
            fc_rows.append(raw.reshape(-1, raw.shape[-1]).cpu())
    fc_art = _solve_weight("fc", model.fc, fc_rows, device=args.fc_device, args=args)
    _apply_dq(model.fc, fc_art[2], device)
    _save_result({"fc": fc_art}, writer, 0)

    target_contexts: list[torch.Tensor] = []
    with torch.no_grad():
        for payload in payloads:
            raw = payload.get("raw_context")
            projected = payload["context"]
            if raw is not None and raw.shape[1] == projected.shape[1]:
                target_contexts.append(model.hidden_norm(model.fc(raw.to(device))))
            else:
                target_contexts.append(projected.to(device))

    total_meta = dict(writer.manifest["tensors"])
    hidden_states = [payload["noise_embedding"].cpu() for payload in payloads]
    for layer_index, layer in enumerate(model.layers):
        manager = CaptureManager(layer, args.exclude, args.group_size)
        next_hidden: list[torch.Tensor] = []
        with manager, torch.no_grad():
            for current_hidden, projected in zip(
                hidden_states, target_contexts, strict=True
            ):
                noise = current_hidden.to(device)
                total_length = int(projected.shape[1] + noise.shape[1])
                position_ids = torch.arange(total_length, device=device).unsqueeze(0)
                position_embeddings = model.rotary_emb(noise, position_ids)
                hidden = layer(
                    hidden_states=noise,
                    target_hidden=projected,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                next_hidden.append(hidden.cpu())
        if not manager.linear_tasks:
            raise RuntimeError(f"layer {layer_index} captured no linear tasks")

        layer_artifacts: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        ] = {}
        for task in manager.linear_tasks:
            name = f"layers.{layer_index}.{task.name}"
            layer_artifacts[name] = _solve_weight(
                name,
                task.module,
                list(manager.stashes[task.stash_key].chunks),
                device=device,
                args=args,
            )
        for task in manager.linear_tasks:
            name = f"layers.{layer_index}.{task.name}"
            _apply_dq(task.module, layer_artifacts[name][2], device)
        _save_result(layer_artifacts, writer, layer_index + 1)
        total_meta.update(writer.manifest["tensors"])

        hidden_states = next_hidden
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        log.info("layer %d complete", layer_index)

    selector_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for hidden in hidden_states:
            selector_rows.append(hidden.reshape(-1, hidden.shape[-1]).cpu())
    selector_art = _solve_weight(
        "candidate_selector.hidden_projection",
        model.candidate_selector.hidden_projection,
        selector_rows,
        device=device,
        args=args,
    )
    _apply_dq(model.candidate_selector.hidden_projection, selector_art[2], device)
    _save_result(
        {"candidate_selector.hidden_projection": selector_art},
        writer,
        len(model.layers) + 1,
    )
    total_meta.update(writer.manifest["tensors"])

    errors = [item["rel_err"] for item in total_meta.values()]
    receipt = {
        "schema_version": 1,
        "cycles": len(payloads),
        "tensors": len(total_meta),
        "elapsed_seconds": time.time() - started,
        "device": device,
        "median_rel_err": float(np.median(errors)),
        "max_rel_err": float(max(errors)),
        "non_claims": [
            "Runtime replay uses exact projected contexts and mask-block embeddings.",
            "The two selector codebooks remain BF16 in this first calibrated artifact.",
            "No acceptance or target-agreement improvement is claimed until benchmarked.",
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv=None) -> int:
    defaults = mode_defaults("mxfp4")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dflash-package", default="/home/kearm/dflash-pytorch")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="mxfp4", choices=("mxfp4", "mxfp8"))
    parser.add_argument("--mxfp-algorithm", default="safe-scale-search")
    parser.add_argument("--clip", default="mse")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fc-device", default="cpu")
    parser.add_argument("--cpu-device", default="cpu")
    parser.add_argument("--cpu-solve-min-in-features", type=int, default=8192)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument(
        "--exclude", default=r"(^|\.)(gate|router)$|shared_expert_gate|e_score"
    )
    parser.set_defaults(bits=defaults[1], group_size=defaults[0])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parser.parse_args(argv)
    receipt = calibrate(args)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
