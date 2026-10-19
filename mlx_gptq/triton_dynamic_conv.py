"""Triton path for DFlash2 grouped dynamic causal convolution."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_dynamic_conv_kernel(
    source_ptr,
    dynamic_ptr,
    base_ptr,
    output_ptr,
    sequence_length,
    source_batch_stride,
    source_sequence_stride,
    dynamic_batch_stride,
    dynamic_sequence_stride,
    hidden_size: tl.constexpr,
    kernel_size: tl.constexpr,
    groups: tl.constexpr,
    group_size: tl.constexpr,
    block_groups: tl.constexpr,
):
    program = tl.program_id(0)
    group_blocks = tl.cdiv(groups, block_groups)
    group_start = (program % group_blocks) * block_groups
    position = (program // group_blocks) % sequence_length
    batch = program // (sequence_length * group_blocks)

    group_offsets = group_start + tl.arange(0, block_groups)
    group_mask = group_offsets < groups
    element_offsets = (
        group_offsets[:, None] * group_size + tl.arange(0, group_size)[None, :]
    )
    element_mask = group_mask[:, None]
    output = tl.zeros((block_groups, group_size), dtype=tl.float32)
    for kernel_index in tl.static_range(kernel_size):
        source_position = position - kernel_index
        source = tl.zeros((block_groups, group_size), dtype=tl.float32)
        if source_position >= 0:
            source_ptr_offset = (
                batch * source_batch_stride
                + source_position * hidden_size
                + element_offsets
            )
            source = tl.load(
                source_ptr + source_ptr_offset, mask=element_mask, other=0.0
            ).to(tl.float32)
        dynamic_offset = (
            batch * dynamic_batch_stride
            + position * kernel_size * groups
            + kernel_index * groups
            + group_offsets
        )
        dynamic = tl.load(dynamic_ptr + dynamic_offset, mask=group_mask, other=0.0).to(
            tl.float32
        )
        base_offset = (
            kernel_index * hidden_size
            + group_offsets[:, None] * group_size
            + tl.arange(0, group_size)[None, :]
        )
        base = tl.load(base_ptr + base_offset, mask=element_mask, other=0.0).to(
            tl.float32
        )
        output += source * (base + dynamic[:, None])
    output_ptr_offset = (
        batch * source_batch_stride + position * hidden_size + element_offsets
    )
    tl.store(
        output_ptr + output_ptr_offset,
        output.to(output_ptr.dtype.element_ty),
        mask=element_mask,
    )


def _launch_convolution(
    source: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    output: torch.Tensor,
) -> None:
    batch, sequence_length, hidden_size = source.shape
    groups = dynamic.shape[-2]
    group_size = hidden_size // groups
    if groups * group_size != hidden_size or base.shape[-1] != hidden_size:
        raise ValueError("invalid grouped dynamic convolution shape")
    kernel_size = base.shape[0]
    block_groups = int(os.environ.get("DFLASH_TRITON_BLOCK_GROUPS", "16"))
    num_warps = int(os.environ.get("DFLASH_TRITON_NUM_WARPS", "1"))
    if block_groups <= 0 or block_groups & (block_groups - 1):
        raise ValueError("DFLASH_TRITON_BLOCK_GROUPS must be a power of two")
    group_blocks = (groups + block_groups - 1) // block_groups
    programs = batch * sequence_length * group_blocks
    _grouped_dynamic_conv_kernel[(programs,)](
        source,
        dynamic,
        base,
        output,
        sequence_length,
        source.stride(0),
        source.stride(1),
        dynamic.stride(0),
        dynamic.stride(1),
        hidden_size,
        kernel_size,
        groups,
        group_size,
        block_groups,
        num_warps=num_warps,
        num_stages=1,
    )


class _TritonGroupedDynamicConv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        dynamic: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        source_contiguous = source.contiguous()
        dynamic_contiguous = dynamic.contiguous()
        base_contiguous = base.contiguous()
        ctx.save_for_backward(source_contiguous, dynamic_contiguous, base_contiguous)
        output = torch.empty_like(source_contiguous)
        _launch_convolution(
            source_contiguous, dynamic_contiguous, base_contiguous, output
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        source, dynamic, base = ctx.saved_tensors
        kernel_size = base.shape[0]
        groups = dynamic.shape[-2]
        group_size = base.shape[-1] // groups
        grouped_shape = (*source.shape[:-1], groups, group_size)
        grad_source = None
        grad_dynamic = None
        grad_base = None
        grouped_gradient = grad_output.reshape(grouped_shape)
        grouped_source = source.reshape(grouped_shape)
        grouped_source_shift = torch.zeros_like(grouped_source)
        if kernel_size > 1:
            grouped_source_shift[:, 1:] = grouped_source[:, :-1]

        if ctx.needs_input_grad[0]:
            grad_source = torch.empty_like(source)
            grouped_base = base[0].reshape(groups, group_size)
            current = grouped_gradient * (grouped_base + dynamic[:, :, 0])
            if kernel_size > 1:
                grouped_base = base[1].reshape(groups, group_size)
                current[:, 1:] += grouped_gradient[:, 1:] * (
                    grouped_base + dynamic[:, 1:, 1]
                )
            grad_source.reshape(grouped_shape).copy_(current)

        if ctx.needs_input_grad[1]:
            grad_dynamic = torch.empty_like(dynamic)
            grad_dynamic[:, :, 0] = (grouped_gradient * grouped_source).sum(
                dim=-1, keepdim=True
            )
            grad_dynamic[:, :, 1] = (grouped_gradient * grouped_source_shift).sum(
                dim=-1, keepdim=True
            )

        if ctx.needs_input_grad[2]:
            grad_base = torch.empty_like(base)
            grad_base[0] = (
                (grouped_gradient * grouped_source).sum(dim=(0, 1)).reshape(-1)
            )
            grad_base[1] = (
                (grouped_gradient * grouped_source_shift).sum(dim=(0, 1)).reshape(-1)
            )

        return grad_source, grad_dynamic, grad_base


def triton_grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Compute the DFlash2 causal convolution with one Triton launch."""
    del group_size
    if base.shape[0] != 2:
        raise ValueError("Triton DFlash2 convolution currently requires kernel size 2")
    if hidden.shape[-1] != base.shape[-1]:
        raise ValueError("hidden and base widths do not match")
    if dynamic.ndim == 4:
        dynamic = dynamic.unsqueeze(-1)
    return _TritonGroupedDynamicConv.apply(hidden, dynamic, base)
