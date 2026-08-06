"""Native MXFP Stage-A receipt and Stage-B injection round trips."""

import json

import numpy as np
import pytest
import torch

from mlx_gptq.artifacts import ArtifactReader, ArtifactWriter
from mlx_gptq.pack import inject, make_predicate
from mlx_gptq.packing import pack_q

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("mode,bits", [("mxfp4", 4), ("mxfp8", 8)])
def test_native_mxfp_artifact_injection(tmp_path, mode, bits):
    calibration = tmp_path / "calibration"
    model = tmp_path / "model"
    model.mkdir()
    key = "model.layers.0.proj.weight"
    base = key.removesuffix(".weight")

    rng = np.random.default_rng(31)
    codes = rng.integers(0, 2**bits, (3, 32), dtype=np.uint8)
    scales = rng.integers(110, 140, (3, 1), dtype=np.uint8)
    meta = {
        key: {
            "mode": mode,
            "bits": bits,
            "group_size": 32,
            "kind": "linear",
        }
    }
    writer = ArtifactWriter(
        str(calibration),
        {
            "model": "synthetic",
            "mode": mode,
            "bits": bits,
            "group_size": 32,
            "storage_dtype": "bfloat16",
        },
    )
    writer.write_layer(
        0,
        {key: (torch.from_numpy(codes), torch.from_numpy(scales), None)},
        meta,
    )

    with open(model / "config.json", "w") as f:
        json.dump(
            {"quantization": {"mode": mode, "bits": bits, "group_size": 32}},
            f,
        )
    mx.save_safetensors(
        str(model / "model.safetensors"),
        {
            key: mx.zeros((3, 32 * bits // 32), dtype=mx.uint32),
            f"{base}.scales": mx.zeros((3, 1), dtype=mx.uint8),
        },
        metadata={"format": "mlx"},
    )

    reader = ArtifactReader(str(calibration))
    q_read, s_read, b_read, info = reader.get(key)
    assert torch.equal(q_read, torch.from_numpy(codes))
    assert torch.equal(s_read, torch.from_numpy(scales))
    assert b_read is None and info["has_biases"] is False

    injected, kept = inject(str(model), reader)
    assert injected == [key]
    assert kept == []
    packed = mx.load(str(model / "model.safetensors"))
    np.testing.assert_array_equal(np.array(packed[key]), pack_q(codes, bits))
    np.testing.assert_array_equal(np.array(packed[f"{base}.scales"]), scales)
    assert f"{base}.biases" not in packed


def test_calibrated_pack_keeps_uncalibrated_endpoints_in_bf16():
    predicate = make_predicate("", None, None, 32, mode="mxfp4")
    assert predicate("model.embed_tokens", object()) is False
    assert predicate("lm_head", object()) is False
    assert predicate("model.layers.0.mlp.down_proj", object()) is True
