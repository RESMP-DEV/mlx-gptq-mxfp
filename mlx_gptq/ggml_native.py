"""Exact native GGML quantization through a local libggml-base library."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GGMLFormat:
    name: str
    type_id: int
    dequantize_symbol: str
    block_size: int
    block_bytes: int

    @property
    def bits_per_weight(self) -> float:
        return self.block_bytes * 8.0 / self.block_size


GGML_FORMATS = {
    "q4_0": GGMLFormat("q4_0", 2, "dequantize_row_q4_0", 32, 18),
    "q4_k": GGMLFormat("q4_k", 12, "dequantize_row_q4_K", 256, 144),
    "iq3_xxs": GGMLFormat(
        "iq3_xxs",
        18,
        "dequantize_row_iq3_xxs",
        256,
        98,
    ),
    "iq4_nl": GGMLFormat(
        "iq4_nl",
        20,
        "dequantize_row_iq4_nl",
        32,
        18,
    ),
    "iq3_s": GGMLFormat("iq3_s", 21, "dequantize_row_iq3_s", 256, 110),
    "iq4_xs": GGMLFormat(
        "iq4_xs",
        23,
        "dequantize_row_iq4_xs",
        256,
        136,
    ),
}


class NativeGGMLQuantizer:
    """Call ggml's production quantizer and reference dequantizer."""

    def __init__(self, library: str) -> None:
        self._library = ctypes.CDLL(library)
        self._library.ggml_quantize_init.argtypes = [ctypes.c_int]
        self._library.ggml_quantize_init.restype = None
        self._library.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._library.ggml_quantize_chunk.restype = ctypes.c_size_t
        self._initialized: set[int] = set()

    def _initialize(self, format_: GGMLFormat) -> None:
        if format_.type_id not in self._initialized:
            self._library.ggml_quantize_init(format_.type_id)
            self._initialized.add(format_.type_id)

    def quantize_blocks(
        self, weight: torch.Tensor, format_: GGMLFormat
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        self._initialize(format_)
        if weight.shape[-1] % format_.block_size:
            raise ValueError(
                f"{format_.name} requires a last dimension divisible by "
                f"{format_.block_size}, got {weight.shape[-1]}"
            )
        rows = int(weight.numel() // weight.shape[-1])
        columns = int(weight.shape[-1])
        source = (
            weight.detach().cpu().reshape(rows, columns).to(torch.float32).contiguous()
        )
        packed = torch.zeros(
            rows,
            columns // format_.block_size * format_.block_bytes,
            dtype=torch.uint8,
        )
        source_pointer = source.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        packed_pointer = packed.numpy().ctypes.data_as(ctypes.c_void_p)
        written = self._library.ggml_quantize_chunk(
            format_.type_id,
            source_pointer,
            packed_pointer,
            0,
            rows,
            columns,
            None,
        )
        expected_bytes = rows * (columns // format_.block_size * format_.block_bytes)
        if written != expected_bytes:
            raise RuntimeError(
                f"ggml_quantize_chunk({format_.name}) wrote {written} bytes, "
                f"expected {expected_bytes}"
            )
        statistics = {
            "format": format_.name,
            "rows": rows,
            "columns": columns,
            "bits_per_weight": format_.bits_per_weight,
            "packed_nonzero_bytes": int(packed.count_nonzero().item()),
        }
        block_count = columns // format_.block_size
        return (
            packed.reshape(rows, block_count, format_.block_bytes),
            statistics,
        )

    def dequantize_blocks(
        self, blocks: torch.Tensor, format_: GGMLFormat
    ) -> torch.Tensor:
        self._initialize(format_)
        if blocks.ndim != 3 or blocks.shape[-1] != format_.block_bytes:
            raise ValueError(
                f"{format_.name} blocks must have shape [rows, blocks, "
                f"{format_.block_bytes}]"
            )
        rows = int(blocks.shape[0])
        columns = int(blocks.shape[1]) * format_.block_size
        packed = blocks.detach().cpu().reshape(rows, -1).contiguous()
        result = torch.empty((rows, columns), dtype=torch.float32)
        dequantize = getattr(self._library, format_.dequantize_symbol)
        dequantize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        dequantize.restype = None
        packed_pointer = packed.numpy().ctypes.data_as(ctypes.c_void_p)
        result_pointer = result.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dequantize(packed_pointer, result_pointer, rows * columns)
        return result

    def quantize_dequantize(
        self, weight: torch.Tensor, format_: GGMLFormat
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        packed, statistics = self.quantize_blocks(weight, format_)
        result = self.dequantize_blocks(packed, format_)
        source = weight.detach().cpu().reshape(result.shape).to(torch.float32)
        error = result - source
        denominator = source.abs().max().clamp_min(1e-12)
        statistics["max_absolute_error"] = float(error.abs().max().item())
        statistics["max_relative_error"] = float(
            (error.abs().max() / denominator).item()
        )
        statistics["relative_l2"] = float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(source).clamp_min(1e-12)
            ).item()
        )
        return result.reshape(weight.shape), statistics
