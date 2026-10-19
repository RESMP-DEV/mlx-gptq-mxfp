import argparse

import torch
from mlx_gptq.artifacts import ArtifactReader
from mlx_gptq.grid import dequantize_mxfp
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
    weights = {
        "fc.weight": torch.randn(8, 64),
        "candidate_selector.successor_codebook.weight": torch.randn(8, 64),
    }
    save_file(weights, str(model / "model.safetensors"))
    output = tmp_path / "calibration"

    result = export(argparse.Namespace(model=model, output=output, qat_source=True))

    assert result["tensors"] == 1
    reader = ArtifactReader(str(output))
    assert reader.tensors["fc"]["mode"] == "mxfp4"
    q, scales, biases, _info = reader.get("fc")
    assert biases is None
    dequantized = dequantize_mxfp(q, scales, "mxfp4")
    assert dequantized.reshape(weights["fc.weight"].shape).shape == (
        weights["fc.weight"].shape
    )
    assert torch.isfinite(dequantized).all()
