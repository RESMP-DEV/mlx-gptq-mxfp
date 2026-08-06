"""LFM ShortConv-aware AWQ scaling on the native MLX MXFP4 grid.

The input-channel scaling is fused across ``operator_norm -> conv.in_proj``.
The output-projection scaling is fused across the C branch of ``in_proj`` and
``conv.out_proj``.  Both transformations preserve the BF16 block function
before quantization.  Only the ShortConv projections are quantized here, so a
partially quantized GPTQ model can be used as the base for a mixed artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx import nn


class MomentCatcher(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.sum_sq = None
        self.rows = 0

    def __call__(self, x, *args, **kwargs):
        axes = tuple(range(x.ndim - 1))
        value = mx.sum(x.astype(mx.float32) ** 2, axis=axes)
        self.sum_sq = value if self.sum_sq is None else self.sum_sq + value
        self.rows += x.size // x.shape[-1]
        return self.module(x, *args, **kwargs)

    def second_moment(self):
        if self.sum_sq is None or not self.rows:
            raise RuntimeError("no activation moments were captured")
        value = self.sum_sq / self.rows
        mx.eval(value)
        return value


def _run_layer(layer, inputs, mask, batch_size):
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        outputs.append(layer(inputs[start : start + batch_size], mask=mask))
        mx.eval(outputs[-1])
    return mx.concatenate(outputs, axis=0)


def _qdq_mxfp4(weight):
    packed = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
    return mx.dequantize(*packed, group_size=32, bits=4, mode="mxfp4")


def _normalized_scale(second_moment, alpha):
    activation = mx.maximum(mx.sqrt(second_moment), 1e-6)
    scale = activation**alpha
    scale = scale / mx.sqrt(mx.maximum(scale.max() * scale.min(), 1e-12))
    return scale


def search_scale(weight, second_moment, grid):
    """Choose an AWQ channel scale using diagonal activation-weighted error."""
    best = None
    for index in range(grid + 1):
        alpha = index / grid
        scale = _normalized_scale(second_moment, alpha)
        dequantized = _qdq_mxfp4(weight * scale[None, :]) / scale[None, :]
        error = (dequantized.astype(mx.float32) - weight.astype(mx.float32)) ** 2
        loss = mx.sum(error * second_moment[None, :]) / weight.size
        mx.eval(loss)
        candidate = (float(loss), alpha, scale)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    mx.eval(best[2])
    return best


def apply_shortconv_scales(layer, in_scale, out_scale):
    """Fuse AWQ scales without changing the unquantized ShortConv function."""
    in_proj = layer.conv.in_proj
    out_proj = layer.conv.out_proj
    hidden = out_proj.weight.shape[-1]

    out_proj.weight = out_proj.weight * out_scale[None, :]
    in_weight = in_proj.weight
    in_weight = mx.concatenate(
        [
            in_weight[:hidden],
            in_weight[hidden : 2 * hidden] / out_scale[:, None],
            in_weight[2 * hidden :],
        ],
        axis=0,
    )
    in_proj.weight = in_weight * in_scale[None, :]
    layer.operator_norm.weight = layer.operator_norm.weight / in_scale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_lfm_shortconv(
    model_path: str,
    calibration: str,
    output_path: str,
    *,
    batch_size: int = 4,
    grid: int = 20,
    num_samples: int | None = None,
):
    import numpy as np
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.utils import load, save

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    model, tokenizer, config = load(model_path, return_config=True, lazy=False)
    tokens_np = np.load(calibration)
    if num_samples is not None:
        tokens_np = tokens_np[:num_samples]
    tokens = mx.array(tokens_np)
    inputs = model.model.embed_tokens(tokens)
    mx.eval(inputs)

    layer_receipts = []
    quant_params = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    quant_config = dict(config["quantization"])

    for layer_index, layer in enumerate(model.layers):
        mask = create_attention_mask(inputs) if layer.is_attention_layer else None
        if layer.is_attention_layer:
            inputs = _run_layer(layer, inputs, mask, batch_size)
            mx.clear_cache()
            continue

        in_proj = layer.conv.in_proj
        out_proj = layer.conv.out_proj
        if isinstance(in_proj, nn.QuantizedLinear) or isinstance(
            out_proj, nn.QuantizedLinear
        ):
            raise TypeError(
                f"layer {layer_index} ShortConv projections must be unquantized"
            )

        in_catcher = MomentCatcher(in_proj)
        out_catcher = MomentCatcher(out_proj)
        layer.conv.in_proj = in_catcher
        layer.conv.out_proj = out_catcher
        teacher_outputs = _run_layer(layer, inputs, mask, batch_size)
        layer.conv.in_proj = in_proj
        layer.conv.out_proj = out_proj

        in_moment = in_catcher.second_moment()
        out_moment = out_catcher.second_moment()

        # First select out_proj's C-branch scale, then select the input scale
        # after accounting for that row transform. Fuse both only afterward.
        out_loss, out_alpha, out_scale = search_scale(
            out_proj.weight, out_moment, grid
        )
        hidden = out_proj.weight.shape[-1]
        in_weight = in_proj.weight
        in_weight = mx.concatenate(
            [
                in_weight[:hidden],
                in_weight[hidden : 2 * hidden] / out_scale[:, None],
                in_weight[2 * hidden :],
            ],
            axis=0,
        )
        in_loss, in_alpha, in_scale = search_scale(in_weight, in_moment, grid)
        apply_shortconv_scales(layer, in_scale, out_scale)

        layer.conv.in_proj = in_proj.to_quantized(
            group_size=32, bits=4, mode="mxfp4"
        )
        layer.conv.out_proj = out_proj.to_quantized(
            group_size=32, bits=4, mode="mxfp4"
        )
        mx.eval(layer.conv.in_proj, layer.conv.out_proj, layer.operator_norm)

        for name in ("in_proj", "out_proj"):
            quant_config[f"model.layers.{layer_index}.conv.{name}"] = quant_params
        layer_receipts.append(
            {
                "layer": layer_index,
                "in_alpha": in_alpha,
                "in_weighted_mse": in_loss,
                "out_alpha": out_alpha,
                "out_weighted_mse": out_loss,
            }
        )
        print(json.dumps(layer_receipts[-1]), flush=True)

        # Standard layer-sequential calibration propagates teacher activations.
        inputs = teacher_outputs
        mx.clear_cache()

    config["quantization"] = quant_config
    config["quantization_config"] = quant_config
    save(output, model_path, model, tokenizer, config)

    weight_files = sorted(output.glob("model*.safetensors"))
    receipt = {
        "algorithm": "lfm-shortconv-awq-channel-scale",
        "base_model": str(Path(model_path).resolve()),
        "calibration": str(Path(calibration).resolve()),
        "calibration_shape": list(tokens_np.shape),
        "batch_size": batch_size,
        "grid": grid,
        "quantization": quant_params,
        "awq_modules": 2 * len(layer_receipts),
        "layers": layer_receipts,
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in weight_files
        ],
    }
    receipt_path = output / "AWQ_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args(argv)
    receipt = quantize_lfm_shortconv(
        args.model,
        args.calibration,
        args.output,
        batch_size=args.batch_size,
        grid=args.grid,
        num_samples=args.num_samples,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
