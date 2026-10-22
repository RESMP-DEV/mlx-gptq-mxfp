from __future__ import annotations

from pathlib import Path

import pytest
from mlx_gptq.ggml_native import GGML_FORMATS, NativeGGMLQuantizer


def test_ggml_storage_geometries_match_their_families() -> None:
    assert GGML_FORMATS["iq4_xs"].bits_per_weight == 4.25
    assert GGML_FORMATS["iq4_nl"].bits_per_weight == 4.5
    assert GGML_FORMATS["q4_0"].bits_per_weight == 4.5
    assert GGML_FORMATS["q4_k"].bits_per_weight == 4.5
    assert GGML_FORMATS["iq3_xxs"].bits_per_weight == 3.0625
    assert GGML_FORMATS["iq3_s"].bits_per_weight == 3.4375


def test_native_quantizer_round_trips_iq4_xs() -> None:
    library = Path(
        "/Users/kearm/Strata/build/llama-tools/bin/libggml-base.0.24.0.dylib"
    )
    if not library.is_file():
        pytest.skip("local native ggml-base library is unavailable")
    import torch

    quantizer = NativeGGMLQuantizer(str(library))
    weight = torch.linspace(-0.2, 0.2, 512, dtype=torch.float32).reshape(2, 256)
    blocks, block_statistics = quantizer.quantize_blocks(weight, GGML_FORMATS["iq4_xs"])
    assert blocks.shape == (2, 1, 136)
    result = quantizer.dequantize_blocks(blocks, GGML_FORMATS["iq4_xs"])
    _, statistics = quantizer.quantize_dequantize(weight, GGML_FORMATS["iq4_xs"])
    assert result.shape == weight.shape
    assert bool(result.isfinite().all())
    assert statistics["bits_per_weight"] == 4.25
    assert block_statistics["bits_per_weight"] == 4.25
    assert 0.0 < float(statistics["relative_l2"]) < 0.2
