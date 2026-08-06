"""Bit-packing q codes into MLX's uint32 layout (numpy, no mlx dependency).

Verified against mx.dequantize: 32/bits consecutive elements per uint32 word,
first element in the least-significant bits. Supported: 2, 4, 8 bits
(3/5/6-bit use a different byte layout in MLX and are not produced here).
"""

from __future__ import annotations

import numpy as np

PACKABLE_BITS = (2, 4, 8)


def pack_q(q: np.ndarray, bits: int) -> np.ndarray:
    """q: uint8 array [..., C] of codes -> uint32 [..., C*bits/32]."""
    if bits not in PACKABLE_BITS:
        raise ValueError(f"cannot pack {bits}-bit for MLX (supported: {PACKABLE_BITS})")
    per = 32 // bits
    *lead, C = q.shape
    assert C % per == 0, f"last dim {C} not divisible by {per}"
    qr = q.reshape(*lead, C // per, per).astype(np.uint32)
    words = np.zeros((*lead, C // per), dtype=np.uint32)
    for i in range(per):
        words |= qr[..., i] << (bits * i)
    return words


def unpack_q(words: np.ndarray, bits: int, out_cols: int) -> np.ndarray:
    per = 32 // bits
    mask = (1 << bits) - 1
    parts = [(words >> (bits * i)) & mask for i in range(per)]
    q = np.stack(parts, axis=-1).reshape(*words.shape[:-1], words.shape[-1] * per)
    return q[..., :out_cols].astype(np.uint8)
