"""Quantization-aware training helpers for the MLX MXFP grid."""

from __future__ import annotations

import torch

from .grid import MX_GROUP_SIZE, choose_mxfp_scale, quantize_mxfp_with_scale


def fake_quantize_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Return an MXFP4 value with a straight-through gradient to weight."""
    if weight.shape[-1] % MX_GROUP_SIZE:
        raise ValueError(f"MXFP4 QAT width must be divisible by {MX_GROUP_SIZE}")
    detached = weight.detach().float()
    grouped = detached.reshape(*detached.shape[:-1], -1, MX_GROUP_SIZE)
    _scale_bytes, scales = choose_mxfp_scale(grouped, "mxfp4")
    _codes, dequantized = quantize_mxfp_with_scale(grouped, scales, "mxfp4")
    dequantized = dequantized.reshape(weight.shape)
    return weight + (dequantized - detached).to(weight.dtype)
