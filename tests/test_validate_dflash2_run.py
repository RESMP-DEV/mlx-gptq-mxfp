import argparse
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tools.validate_dflash2_run import validate


def _run(
    tmp_path: Path, *, zero_b: bool = False, qat: bool = False
) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    adapters = {}
    for index in range(46):
        value = 0.0 if zero_b and index == 0 else 0.125
        adapters[f"layers.{index}.b"] = torch.tensor([value])
    save_file(adapters, str(run / "lora_adapters.safetensors"))
    (run / "receipt.json").write_text(
        json.dumps(
            {
                "steps": 128,
                "replay_fidelity_cosine": 0.99,
                "qat_mxfp4": qat,
            }
        )
    )
    if qat:
        grids = {}
        for index in range(47):
            grids[f"tensor.{index}.q"] = torch.zeros(1, 32, dtype=torch.uint8)
            grids[f"tensor.{index}.scales"] = torch.zeros(1, 1, dtype=torch.uint8)
        save_file(grids, str(run / "qat_mxfp4.safetensors"))
    return run


def test_validate_accepts_nonzero_lora_updates(tmp_path: Path):
    result = validate(
        argparse.Namespace(
            run=_run(tmp_path),
            expected_steps=128,
            fidelity_threshold=0.97,
            qat=False,
        )
    )

    assert result["lora_b_tensors"] == 46
    assert result["min_b_max_abs"] == 0.125


def test_validate_rejects_zero_lora_update(tmp_path: Path):
    with pytest.raises(RuntimeError, match="did not update"):
        validate(
            argparse.Namespace(
                run=_run(tmp_path, zero_b=True),
                expected_steps=128,
                fidelity_threshold=0.97,
                qat=False,
            )
        )


def test_validate_counts_qat_prescribed_grids(tmp_path: Path):
    result = validate(
        argparse.Namespace(
            run=_run(tmp_path, qat=True),
            expected_steps=128,
            fidelity_threshold=0.97,
            qat=True,
        )
    )

    assert result["qat_grid_pairs"] == 47
