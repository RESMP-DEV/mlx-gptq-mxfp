"""Native MXFP4 matmul for DFlash2 shapes on SM120.

The kernel consumes exactly what the DFlash2 packer emits: packed E2M1 codes
in uint32 words and one E8M0 scale byte per 32 values. The activation operand
is quantized to the same MXFP4 format.

Scale bytes are converted into the preshuffled layout that
``tl.dot_scaled`` expects, so the kernel lowers to the native
``mma.sync...kind::mxf4nvf4`` instruction on SM120 rather than a BF16
decomposition. Run it on a CUDA host with a compute capability 12.0 device.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

try:
    from tools.mxfp4_kernel.dflash2_shapes import dominant_shapes
    from tools.mxfp4_kernel.e2m1_packing import (
        pack_e2m1_bytes,
        unpack_e2m1,
    )
except ModuleNotFoundError:
    from dflash2_shapes import dominant_shapes
    from e2m1_packing import pack_e2m1_bytes, unpack_e2m1

E2M1_MAX = 6.0
E4M3_MAX = 448.0
VEC_SIZE = 32
VEC_SIZE_TL = tl.constexpr(32)
TILE_M = 128
TILE_N = 128
TILE_K = 128
ACT_MAX_GROUPS = 64
ACT_PACK_BYTES = 2048
WEIGHT_PACK_WORDS = 512
SCALE_TILE_ROWS = 128


@triton.jit
def _repack_e2m1_words_kernel(
    words_ptr,
    packed_ptr,
    words,
    word_stride,
    BLOCK_WORDS: tl.constexpr,
):
    """Convert packed uint32 E2M1 words to Triton's byte layout."""
    row = tl.program_id(0)
    word_block = tl.program_id(1)
    words_off = word_block * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    word_mask = words_off < words
    word_values = tl.load(
        words_ptr + row * word_stride + words_off, mask=word_mask, other=0
    )
    byte_off = tl.arange(0, 4)[None, :]
    low = (word_values[:, None] >> (byte_off * 8)) & 0xF
    high = (word_values[:, None] >> (byte_off * 8 + 4)) & 0xF
    packed = (low | (high << 4)).to(tl.uint8)
    tl.store(
        packed_ptr + row * words * 4 + words_off[:, None] * 4 + byte_off,
        packed,
        mask=word_mask[:, None],
    )


def repack_e2m1_words_to_bytes(words: torch.Tensor) -> torch.Tensor:
    """Repack uint32 E2M1 weights without large int64 temporaries."""
    if words.ndim != 2:
        raise ValueError("packed words must be a rank-2 tensor")
    if words.dtype != torch.uint32 or not words.is_cuda:
        raise ValueError("packed words must be CUDA uint32")
    source = words.contiguous()
    rows, word_count = source.shape
    packed = torch.empty(
        (rows, word_count * 4), device=source.device, dtype=torch.uint8
    )
    grid = (rows, triton.cdiv(word_count, WEIGHT_PACK_WORDS))
    _repack_e2m1_words_kernel[grid](
        source,
        packed,
        word_count,
        source.stride(0),
        BLOCK_WORDS=WEIGHT_PACK_WORDS,
    )
    return packed


def e8m0_to_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 exponent bytes to float32 power-of-two scales."""
    exponent = scale_bytes.to(torch.float32) - 127.0
    return torch.pow(torch.tensor(2.0, device=scale_bytes.device), exponent)


def preshuffle_scales(scale_bytes: torch.Tensor) -> torch.Tensor:
    """Convert row-major E8M0 scales into the hardware block-scaled layout.

    The input has one byte per 32 values per row. Triton expects
    ``(rows // 128, groups // 4, 32, 4, 4)``. Kernel loading and reshape
    produce logical row ``128*block + m4*32 + row32`` and group
    ``4*block + k4``.
    """
    if TILE_M != SCALE_TILE_ROWS:
        raise RuntimeError("scale layout requires a 128-row tile")
    rows, groups = scale_bytes.shape
    if rows % SCALE_TILE_ROWS or groups % 4:
        raise ValueError("scale rows must be a multiple of 128 and groups of 4")
    logical = scale_bytes.reshape(
        rows // SCALE_TILE_ROWS, 4, 32, groups // 4, 4
    )
    return logical.permute(0, 3, 2, 1, 4).contiguous()


def quantize_e2m1_codes(normalized: torch.Tensor) -> torch.Tensor:
    """Round normalized values to uint8 E2M1 codes."""
    magnitude = torch.where(
        normalized.abs() > 5.0,
        torch.full_like(normalized, 7.0),
        torch.where(
            normalized.abs() >= 3.5,
            torch.full_like(normalized, 6.0),
            torch.where(
                normalized.abs() > 2.5,
                torch.full_like(normalized, 5.0),
                torch.where(
                    normalized.abs() >= 1.75,
                    torch.full_like(normalized, 4.0),
                    torch.where(
                        normalized.abs() > 1.25,
                        torch.full_like(normalized, 3.0),
                        torch.where(
                            normalized.abs() >= 0.75,
                            torch.full_like(normalized, 2.0),
                            torch.where(
                                normalized.abs() > 0.25,
                                torch.full_like(normalized, 1.0),
                                torch.zeros_like(normalized),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    sign = torch.where(normalized < 0, 8.0, 0.0)
    return (magnitude + sign).to(torch.uint8)


def quantize_activation_mxfp4(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-scale activations to MXFP4 with E8M0 group scales."""
    grouped = activations.reshape(*activations.shape[:-1], -1, VEC_SIZE).float()
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale_bytes, scales = _activation_e8m0_scales(max_abs)
    codes = quantize_e2m1_codes(grouped / scales)
    return codes, scale_bytes


def _activation_e8m0_scales(
    max_abs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E8M0 bytes and float scales for per-group maxima."""
    safe = torch.where(max_abs == 0, torch.full_like(max_abs, E2M1_MAX), max_abs)
    exponent = torch.round(torch.log2(safe / E2M1_MAX)).clamp(-127, 127)
    exponent = torch.where(max_abs == 0, torch.zeros_like(exponent), exponent)
    scale_bytes = (exponent + 127).to(torch.uint8)
    scales = torch.pow(
        torch.tensor(2.0, device=max_abs.device), exponent.float()
    )
    return scale_bytes, scales


@triton.jit
def _activation_group_max_kernel(
    activations_ptr,
    max_abs_ptr,
    active_rows,
    activation_stride,
    groups,
    BLOCK_GROUPS: tl.constexpr,
):
    """Reduce one row tile to one maximum per 32-value group."""
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    groups_off = group_block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    values_off = tl.arange(0, 32)
    group_mask = groups_off < groups
    value_mask = (row < active_rows) & group_mask[:, None]
    offsets = (
        row * activation_stride
        + groups_off[:, None] * 32
        + values_off[None, :]
    )
    values = tl.load(activations_ptr + offsets, mask=value_mask, other=0.0)
    max_abs = tl.max(tl.abs(values), axis=1)
    tl.store(max_abs_ptr + row * groups + groups_off, max_abs, mask=group_mask)


@triton.jit
def _e2m1_code(normalized):
    """Return one signed E2M1 code for a normalized Triton value."""
    magnitude = tl.abs(normalized)
    code = tl.where(
        magnitude > 5.0,
        7,
        tl.where(
            magnitude >= 3.5,
            6,
            tl.where(
                magnitude > 2.5,
                5,
                tl.where(
                    magnitude >= 1.75,
                    4,
                    tl.where(
                        magnitude > 1.25,
                        3,
                        tl.where(
                            magnitude >= 0.75,
                            2,
                            tl.where(magnitude > 0.25, 1, 0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return code + tl.where(normalized < 0.0, 8, 0)


@triton.jit
def _activation_quantize_pack_kernel(
    activations_ptr,
    scales_ptr,
    packed_ptr,
    active_rows,
    activation_stride,
    columns,
    groups,
    BLOCK_BYTES: tl.constexpr,
):
    """Quantize and pack activation values in Triton's FP4 byte layout."""
    row = tl.program_id(0)
    byte_block = tl.program_id(1)
    bytes_off = byte_block * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    low_off = bytes_off * 2
    high_off = low_off + 1
    low_mask = (row < active_rows) & (low_off < columns)
    high_mask = (row < active_rows) & (high_off < columns)
    low_values = tl.load(
        activations_ptr + row * activation_stride + low_off,
        mask=low_mask,
        other=0.0,
    )
    high_values = tl.load(
        activations_ptr + row * activation_stride + high_off,
        mask=high_mask,
        other=0.0,
    )
    low_scales = tl.load(
        scales_ptr + row * groups + (low_off // 32),
        mask=low_mask,
        other=1.0,
    )
    high_scales = tl.load(
        scales_ptr + row * groups + (high_off // 32),
        mask=high_mask,
        other=1.0,
    )
    low = _e2m1_code(low_values / low_scales).to(tl.uint8)
    high = _e2m1_code(high_values / high_scales).to(tl.uint8)
    packed = low | (high << 4)
    tl.store(
        packed_ptr + row * (columns // 2) + bytes_off,
        packed,
        mask=bytes_off < (columns // 2),
    )


def dequantize_mxfp4(
    codes: torch.Tensor, scale_bytes: torch.Tensor
) -> torch.Tensor:
    """Dequantize packed E2M1 codes with E8M0 scales to float32."""
    lut = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        device=codes.device,
    )
    grouped = codes.reshape(*codes.shape[:-1], -1, 8).to(torch.long)
    values = lut[grouped]
    values = values.reshape(codes.shape)
    scales = e8m0_to_scale(scale_bytes)
    return values.reshape(*codes.shape[:-1], -1, VEC_SIZE) * scales.unsqueeze(-1)


def dequantize_mxfp8(
    codes: torch.Tensor, scale_bytes: torch.Tensor
) -> torch.Tensor:
    """Dequantize MXFP8 (E4M3) codes with E8M0 scales to float32."""
    grouped = codes.reshape(*codes.shape[:-1], -1, VEC_SIZE).contiguous()
    values = grouped.view(torch.float8_e4m3fn).float()
    scales = e8m0_to_scale(scale_bytes)
    return values * scales.unsqueeze(-1)


@triton.jit
def _mxfp4_matmul_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A @ B with native block-scaled FP4 operands."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_packed_k = tl.arange(0, BLOCK_K // 2)
    scale_values = tl.arange(0, 32 * 4 * 4)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, num_k_blocks):
        a_offs = (
            offs_am[:, None] * stride_am
            + (k * (BLOCK_K // 2) + offs_packed_k)[None, :] * stride_ak
        )
        b_offs = (
            offs_bn[:, None] * stride_bn
            + (k * (BLOCK_K // 2) + offs_packed_k)[None, :] * stride_bk
        )
        a = tl.load(a_ptr + a_offs)
        b = tl.trans(tl.load(b_ptr + b_offs))

        a_scale_base = (pid_m * num_k_blocks + k) * (32 * 4 * 4)
        b_scale_base = (pid_n * num_k_blocks + k) * (32 * 4 * 4)
        a_scale = tl.load(a_scale_ptr + a_scale_base + scale_values)
        b_scale = tl.load(b_scale_ptr + b_scale_base + scale_values)
        a_scale = a_scale.reshape(32, 4, 4).trans(1, 0, 2)
        b_scale = b_scale.reshape(32, 4, 4).trans(1, 0, 2)
        a_scale = a_scale.reshape(BLOCK_M, BLOCK_K // VEC_SIZE_TL)
        b_scale = b_scale.reshape(BLOCK_N, BLOCK_K // VEC_SIZE_TL)

        accumulator = tl.dot_scaled(
            a, a_scale, "e2m1", b, b_scale, "e2m1", accumulator
        )

    c_offs = offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptr + c_offs, accumulator.to(tl.bfloat16))


def prepare_mxfp4_operands(
    activations: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Pad, quantize, and pack operands for repeated kernel launches."""
    rows = activations.shape[0]
    outputs = weight_codes.shape[0]
    columns = weight_codes.shape[1] * 8
    if outputs % TILE_N:
        raise ValueError("output features must be divisible by 128")
    if columns % TILE_K:
        raise ValueError("input features must be divisible by 128")
    padded_rows = ((rows + TILE_M - 1) // TILE_M) * TILE_M
    padded_activations = torch.zeros(
        (padded_rows, columns), device=activations.device, dtype=torch.float32
    )
    padded_activations[:rows] = activations

    a_codes, a_scales = quantize_activation_mxfp4(padded_activations)
    a_flat = pack_e2m1_bytes(a_codes.reshape(padded_rows, columns))
    a_scale_flat = preshuffle_scales(
        a_scales.reshape(padded_rows, columns // VEC_SIZE)
    )
    b_codes = unpack_e2m1(weight_codes, columns)
    b_flat = pack_e2m1_bytes(b_codes)
    b_scale_flat = preshuffle_scales(weight_scales)
    return a_flat, a_scale_flat, b_flat, b_scale_flat, padded_rows, outputs, columns


def prepare_activation_mxfp4_fast(
    activations: torch.Tensor,
    padded_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Produce packed activations with fused Triton max and pack kernels.

    PyTorch still computes E8M0 exponents from the small max tensor. This
    preserves the reference rounding decision while removing the large
    intermediate code tensors from the preparation path.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be a rank-2 tensor")
    if not torch.is_floating_point(activations) or not activations.is_cuda:
        raise ValueError("activations must be CUDA floating point")
    rows, columns = activations.shape
    if columns % TILE_K:
        raise ValueError("input features must be divisible by 128")
    if padded_rows is None:
        padded_rows = ((rows + TILE_M - 1) // TILE_M) * TILE_M
    if not rows <= padded_rows or padded_rows % TILE_M:
        raise ValueError("padded_rows must be a multiple of 128 and cover rows")

    source = activations.to(torch.float32).contiguous()
    groups = columns // VEC_SIZE
    max_abs = torch.empty(
        (padded_rows, groups), device=source.device, dtype=torch.float32
    )
    max_grid = (
        padded_rows,
        triton.cdiv(groups, ACT_MAX_GROUPS),
    )
    _activation_group_max_kernel[max_grid](
        source,
        max_abs,
        rows,
        source.stride(0),
        groups,
        BLOCK_GROUPS=ACT_MAX_GROUPS,
    )
    scale_bytes, scales = _activation_e8m0_scales(max_abs.unsqueeze(-1))
    a_scale_flat = preshuffle_scales(scale_bytes.reshape(padded_rows, groups))

    a_flat = torch.empty(
        (padded_rows, columns // 2), device=source.device, dtype=torch.uint8
    )
    pack_grid = (
        padded_rows,
        triton.cdiv(columns // 2, ACT_PACK_BYTES),
    )
    _activation_quantize_pack_kernel[pack_grid](
        source,
        scales,
        a_flat,
        rows,
        source.stride(0),
        columns,
        groups,
        BLOCK_BYTES=ACT_PACK_BYTES,
    )
    return a_flat, a_scale_flat, padded_rows, columns


def prepare_mxfp4_operands_fast(
    activations: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Prepare both operands with the fused activation path."""
    outputs = weight_codes.shape[0]
    columns = weight_codes.shape[1] * 8
    if (
        not activations.is_cuda
        or weight_codes.device != activations.device
        or weight_scales.device != activations.device
    ):
        raise ValueError("all operands must be on one CUDA device")
    if weight_codes.dtype != torch.uint32:
        raise ValueError("weight codes must be uint32")
    if weight_scales.dtype != torch.uint8:
        raise ValueError("weight scales must be uint8")
    if outputs % TILE_N or columns % TILE_K:
        raise ValueError("DFlash2 projection dimensions must be divisible by 128")
    if activations.ndim != 2 or activations.shape[1] != columns:
        raise ValueError("activation columns must match the packed weight")
    if weight_scales.shape != (outputs, columns // VEC_SIZE):
        raise ValueError("weight scale shape does not match the packed weight")
    a_flat, a_scale_flat, padded_rows, columns = prepare_activation_mxfp4_fast(
        activations
    )
    b_flat = repack_e2m1_words_to_bytes(weight_codes)
    b_scale_flat = preshuffle_scales(weight_scales)
    return a_flat, a_scale_flat, b_flat, b_scale_flat, padded_rows, outputs, columns


def mxfp4_matmul_prepared(
    prepared: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
    ],
) -> torch.Tensor:
    """Run the native matmul on prepared operands."""
    a_flat, a_scale_flat, b_flat, b_scale_flat, padded_rows, outputs, columns = prepared
    output = torch.empty(
        (padded_rows, outputs), device=a_flat.device, dtype=torch.bfloat16
    )
    grid = (triton.cdiv(padded_rows, TILE_M) * triton.cdiv(outputs, TILE_N),)
    _mxfp4_matmul_kernel[grid](
        a_flat,
        a_scale_flat,
        b_flat,
        b_scale_flat,
        output,
        padded_rows,
        outputs,
        columns,
        a_flat.stride(0),
        a_flat.stride(1),
        b_flat.stride(0),
        b_flat.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=TILE_M,
        BLOCK_N=TILE_N,
        BLOCK_K=TILE_K,
    )
    return output


def mxfp4_matmul(
    activations: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """Run the native block-scaled matmul for one DFlash2 projection."""
    prepared = prepare_mxfp4_operands_fast(
        activations, weight_codes, weight_scales
    )
    return mxfp4_matmul_prepared(prepared)[: activations.shape[0]]

def reference_matmul(
    activations: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """BF16 dequantized reference for the same packed weight."""
    a_codes, a_scales = quantize_activation_mxfp4(activations)
    a_dequant = dequantize_mxfp4(a_codes, a_scales).reshape(activations.shape)
    b_dequant = dequantize_mxfp4(
        unpack_e2m1(weight_codes, weight_codes.shape[1] * 8), weight_scales
    ).reshape(weight_codes.shape[0], -1)
    a_bf16 = a_dequant.to(torch.bfloat16)
    b_bf16 = b_dequant.to(torch.bfloat16)
    return torch.nn.functional.linear(a_bf16, b_bf16)


def inspect_native_ptx() -> dict[str, object]:
    """Inspect compiled PTX and require the native block-scaled MMA."""
    cache_tuple = _mxfp4_matmul_kernel.device_caches[torch.cuda.current_device()]
    kernels = list(cache_tuple[0].values())
    markers = []
    for kernel in kernels:
        ptx = kernel.asm.get("ptx", "")
        markers.append(
            {
                "native_mxf4nvf4": "kind::mxf4nvf4" in ptx,
                "native_ue8m0": "ue8m0" in ptx,
                "bf16_mma": "mma.sync" in ptx and "bf16" in ptx,
                "ptx_bytes": len(ptx),
            }
        )
    native = any(item["native_mxf4nvf4"] for item in markers)
    if not native:
        raise RuntimeError("Triton compiled a non-native block-scaled path")
    return {"kernels": len(markers), "ptx_markers": markers}


def _timed(operation, warmup: int, repeats: int) -> tuple[list[float], torch.Tensor]:
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        output = operation()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times, output


def _make_inputs(
    rows: int, outputs: int, columns: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a packed MXFP4 weight and a matching activation."""
    weight_codes = (
        torch.randint(
            0,
            256,
            (outputs, columns // 8, 4),
            device=device,
            dtype=torch.uint8,
        )
        .view(torch.uint32)
        .reshape(outputs, columns // 8)
    )
    weight_scales = torch.randint(
        120,
        132,
        (outputs, columns // VEC_SIZE),
        device=device,
        dtype=torch.uint8,
    )
    activations = torch.randn(rows, columns, device=device, dtype=torch.float32) * 0.5
    return activations, weight_codes, weight_scales


def run_shape(
    name: str,
    rows: int,
    outputs: int,
    columns: int,
    device: str,
    warmup: int,
    repeats: int,
    max_rel_error: float = 0.05,
) -> dict[str, Any]:
    """Benchmark and verify one shape."""
    activations, weight_codes, weight_scales = _make_inputs(
        rows, outputs, columns, device
    )
    torch.cuda.synchronize()
    cold_started = torch.cuda.Event(enable_timing=True)
    cold_finished = torch.cuda.Event(enable_timing=True)
    cold_started.record()
    cold_prepared = prepare_mxfp4_operands_fast(
        activations, weight_codes, weight_scales
    )
    cold_finished.record()
    torch.cuda.synchronize()
    prepare_cold_ms = cold_started.elapsed_time(cold_finished)
    del cold_prepared

    prepare_started = torch.cuda.Event(enable_timing=True)
    prepare_finished = torch.cuda.Event(enable_timing=True)
    prepare_started.record()
    prepared = prepare_mxfp4_operands_fast(
        activations, weight_codes, weight_scales
    )
    prepare_finished.record()
    torch.cuda.synchronize()
    prepare_ms = prepare_started.elapsed_time(prepare_finished)
    kernel_times, kernel_output_padded = _timed(
        lambda: mxfp4_matmul_prepared(prepared), warmup, repeats
    )
    kernel_output = kernel_output_padded[:rows]
    reference = reference_matmul(activations, weight_codes, weight_scales)
    error = (kernel_output.float() - reference.float()).abs()
    reference_abs = reference.float().abs()
    reference_max = reference_abs.max().clamp(min=1e-6)
    element_relative = error / reference_abs.clamp_min(1e-6)
    tolerance = reference_max * 0.01 + max_rel_error * reference_abs
    error_ratio = error / tolerance
    median_ms = statistics.median(kernel_times)
    global_relative = float(error.max() / reference_max)
    relative = float(element_relative.max())
    maximum_error_ratio = float(error_ratio.max())
    return {
        "name": name,
        "rows": rows,
        "outputs": outputs,
        "columns": columns,
        "median_ms": median_ms,
        "prepare_cold_ms": prepare_cold_ms,
        "prepare_ms": prepare_ms,
        "kernel_logical_tflops_per_second": (
            2 * rows * outputs * columns / (median_ms / 1000) / 1e12
        ),
        "max_abs_error": float(error.max()),
        "max_global_rel_error": global_relative,
        "max_rel_error": relative,
        "max_error_tolerance_ratio": maximum_error_ratio,
        "finite": bool(torch.isfinite(kernel_output.float()).all()),
        "passed": bool(maximum_error_ratio <= 1.0) and bool(
            torch.isfinite(kernel_output.float()).all()
        ),
    }


def native_mma_markers() -> dict[str, bool]:
    """Report the MMA instruction families in the compiled kernel PTX."""
    cache = _mxfp4_matmul_kernel.device_caches
    if not cache:
        return {"ptx_available": False}
    entry = cache[torch.cuda.current_device()]
    kernels = entry[0]
    markers = {
        "ptx_available": bool(kernels),
        "mxf4nvf4": False,
        "bf16_mma": False,
        "fp8_mma": False,
    }
    for key, kernel in kernels.items():
        ptx = kernel.asm.get("ptx", "")
        markers["mxf4nvf4"] |= "kind::mxf4nvf4" in ptx
        markers["bf16_mma"] |= "kind::mxf4" not in ptx and "bf16" in ptx
        markers["fp8_mma"] |= "e4m3" in ptx and "kind::mxf4nvf4" not in ptx
    return markers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("mxfp4-matmul-results.json")
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    parser.add_argument("--max-rel-error", type=float, default=0.05)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    capability = (properties.major, properties.minor)
    shapes = dominant_shapes()
    if arguments.rows:
        shapes = [row for row in shapes if row[1] in set(arguments.rows)]

    results = []
    for name, rows, outputs, columns in shapes:
        result = run_shape(
            name,
            rows,
            outputs,
            columns,
            "cuda",
            arguments.warmup,
            arguments.repeats,
            arguments.max_rel_error,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    markers = native_mma_markers()

    receipt = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "operator": "Codex",
        "device": {
            "name": properties.name,
            "capability": list(capability),
            "total_memory": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "kernel": "triton tl.dot_scaled mxfp4 by mxfp4 with fused activation pack",
        "native_mma_markers": markers,
        "max_rel_error_threshold": arguments.max_rel_error,
        "results": results,
        "non_claims": [
            "Both operands are block-scaled MXFP4; mixed MXFP8 by MXFP4 "
            "compiled to BF16 emulation in the live probe.",
            "The gate is per-element relative error plus an absolute floor of "
            "one percent of the global reference maximum.",
            "Synthetic operands prove kernel numerics and throughput, not "
            "DFlash2 model quality.",
            "Isolated GEMM latency does not predict end-to-end training or "
            "serving throughput.",
            "No DFlash2 artifact is packed, run, or accepted by this benchmark.",
        ],
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    failed = [row["name"] for row in results if not row["passed"]]
    if failed:
        print(f"numerical gate failed for: {failed}")
        return 1
    if not markers.get("mxf4nvf4"):
        print("native MXFP4 MMA instruction was not found in the compiled PTX")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
