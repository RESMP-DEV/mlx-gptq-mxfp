import argparse

import pytest
import torch
from mlx_gptq.artifacts import ArtifactReader
from mlx_gptq.grid import (
    choose_mxfp_scale,
    dequantize_mxfp,
    quantize_mxfp_with_scale,
    rtn_quantize_mxfp,
)
from safetensors.torch import save_file
from tools.export_rtn_mxfp4 import export, target_tensor_names


def test_target_tensor_names_selects_dflash2_linears():
    weights = {
        "fc.weight": torch.zeros(8, 64),
        "candidate_selector.hidden_projection.weight": torch.zeros(4, 64),
        "layers.0.self_attn.q_proj.weight": torch.zeros(8, 64),
        "layers.1.mlp.down_proj.weight": torch.zeros(8, 64),
        "candidate_selector.successor_codebook.weight": torch.zeros(8, 64),
        "layers.0.input_layernorm.weight": torch.zeros(64),
        "layers.0.attention_conv.base_kernel": torch.zeros(2, 2, 64),
    }

    assert set(target_tensor_names(weights)) == {
        "fc",
        "candidate_selector.hidden_projection",
        "layers.0.self_attn.q_proj",
        "layers.1.mlp.down_proj",
    }


def test_export_rtn_mxfp4_round_trips(tmp_path):
    torch.manual_seed(3)
    model = tmp_path / "model"
    model.mkdir()
    raw = torch.randn(8, 64)
    raw[0, 0] = 2.2
    grouped = raw.reshape(8, 2, 32)
    scale_bytes, scales = choose_mxfp_scale(grouped, "mxfp4")
    codes, effective = quantize_mxfp_with_scale(grouped, scales, "mxfp4")
    weights = {
        "fc.weight": effective.reshape(raw.shape),
        "candidate_selector.successor_codebook.weight": torch.randn(8, 64),
    }
    save_file(
        {
            "fc.q": codes.reshape(raw.shape),
            "fc.scales": scale_bytes.reshape(8, 2),
        },
        str(model / "qat_mxfp4.safetensors"),
    )
    save_file(weights, str(model / "model.safetensors"))
    output = tmp_path / "calibration"

    result = export(argparse.Namespace(model=model, output=output, qat_source=True))

    assert result["tensors"] == 1
    assert result["source"] == "qat-prescribed-grid"
    reader = ArtifactReader(str(output))
    assert reader.tensors["fc"]["mode"] == "mxfp4"
    q, scales, biases, _info = reader.get("fc")
    assert biases is None
    dequantized = dequantize_mxfp(q, scales, "mxfp4")
    assert dequantized.reshape(weights["fc.weight"].shape).shape == (
        weights["fc.weight"].shape
    )
    assert torch.isfinite(dequantized).all()
    assert torch.equal(q, codes.reshape(raw.shape))
    assert torch.equal(scales, scale_bytes.reshape(8, 2))
    naive = rtn_quantize_mxfp(weights["fc.weight"][None], "mxfp4")
    assert not torch.equal(scales, naive[1][0])


def test_qat_export_rejects_missing_prescribed_grid(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    save_file({"fc.weight": torch.randn(8, 64)}, str(model / "model.safetensors"))

    with pytest.raises(FileNotFoundError, match="qat_mxfp4"):
        export(
            argparse.Namespace(
                model=model, output=tmp_path / "out", qat_source=True
            )
        )
