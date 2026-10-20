"""Pure PyTorch E2M1 layout helpers for the SM120 harness."""

from __future__ import annotations

import torch


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 E2M1 codes into uint32 words, LSB first."""
    if codes.shape[-1] % 8:
        raise ValueError("column count must be divisible by 8 for packing")
    rows, columns = codes.shape
    words = codes.reshape(rows, columns // 8, 8).to(torch.int64)
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * 4
    packed = (words << shifts).sum(dim=-1)
    return packed.to(torch.uint32)


def pack_e2m1_bytes(codes: torch.Tensor) -> torch.Tensor:
    """Pack adjacent E2M1 codes into Triton's byte layout."""
    if codes.shape[-1] % 2:
        raise ValueError("column count must be divisible by 2")
    grouped = codes.reshape(*codes.shape[:-1], -1, 2).to(torch.int64)
    packed = grouped[..., 0] | (grouped[..., 1] << 4)
    return packed.to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor, columns: int) -> torch.Tensor:
    """Unpack uint32 E2M1 words into uint8 codes."""
    if packed.ndim != 2 or columns % 8:
        raise ValueError("packed words must be rank 2 with columns divisible by 8")
    if packed.shape[1] * 8 != columns:
        raise ValueError("column count does not match the packed words")
    rows = packed.shape[0]
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    words = packed.to(torch.int64).unsqueeze(-1)
    codes = (words >> shifts) & 0xF
    return codes.reshape(rows, columns).to(torch.uint8)
