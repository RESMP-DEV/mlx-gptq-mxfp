import numpy as np
import pytest
from mlx_gptq.packing import pack_mxfp4, unpack_q


def test_pack_mxfp4_is_lsb_first_and_byte_ordered():
    codes = np.arange(16, dtype=np.uint8)
    packed = pack_mxfp4(codes)
    assert packed.dtype == np.uint32
    assert packed.shape == (2,)
    assert unpack_q(packed, 4, 16).reshape(-1).tolist() == list(range(16))


def test_pack_mxfp4_rejects_unaligned_columns():
    with pytest.raises(AssertionError):
        pack_mxfp4(np.zeros(4, dtype=np.uint8))


def test_pack_reads_config_from_explicit_root(tmp_path):
    """A weights-only training run must pack with the base model config."""
    import argparse
    import json

    import torch
    from mlx_gptq.artifacts import ArtifactWriter
    from mlx_gptq.dflash2_pack import pack
    from mlx_gptq.grid import rtn_quantize_mxfp
    from safetensors.torch import save_file

    config_root = tmp_path / "base"
    config_root.mkdir()
    (config_root / "config.json").write_text(json.dumps({"model_type": "draft"}))
    run = tmp_path / "run"
    run.mkdir()
    weight = torch.randn(8, 64)
    save_file({"fc.weight": weight}, str(run / "model.safetensors"))

    calibration = tmp_path / "calibration"
    writer = ArtifactWriter(
        str(calibration),
        {
            "mode": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "storage_dtype": "bfloat16",
        },
    )
    q, scales, _biases, _dq = rtn_quantize_mxfp(weight[None], "mxfp4")
    writer.write_layer(
        0,
        {"fc": (q[0], scales[0], None)},
        {
            "fc": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
                "kind": "dflash2-linear",
                "out_features": 8,
                "in_features": 64,
            }
        },
    )

    output = tmp_path / "packed"
    receipt = pack(
        argparse.Namespace(
            model=run, calibration=calibration, output=output, config_root=config_root
        )
    )

    assert receipt["config_source"] == str(config_root)
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "draft"
    assert config["quantization"]["mode"] == "mxfp4"


def test_pack_rejects_missing_config():
    import pytest
    from mlx_gptq.dflash2_pack import _config_source

    with pytest.raises(FileNotFoundError):
        _config_source(__import__("pathlib").Path("/nonexistent-model"), None)
