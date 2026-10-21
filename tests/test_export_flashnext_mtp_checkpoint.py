from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_SCRIPT = Path(__file__).parents[1] / "scripts" / "export_flashnext_mtp_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location(
    "export_flashnext_mtp_checkpoint", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def test_export_selects_mtp_and_shared_tensors_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "compact"
    source.mkdir()
    save_file(
        {
            "mtp.layers.0.fc_embedding.weight": torch.ones((2, 2)),
            "mtp.shared.weight": torch.ones((2, 2)),
            "model.language_model.embed_tokens.weight": torch.ones((2, 2)),
            "lm_head.weight": torch.ones((2, 2)),
            "model.language_model.layers.0.weight": torch.ones((2, 2)),
        },
        str(source / "model-00001-of-00001.safetensors"),
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 64},
                "weight_map": {
                    name: "model-00001-of-00001.safetensors"
                    for name in (
                        "mtp.layers.0.fc_embedding.weight",
                        "mtp.shared.weight",
                        "model.language_model.embed_tokens.weight",
                        "lm_head.weight",
                        "model.language_model.layers.0.weight",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")

    receipt = exporter.export_checkpoint(source, output)

    assert receipt["tensor_count"] == 4
    assert receipt["shard_count"] == 1
    assert (
        receipt["total_bytes"]
        == (output / "model-00001-of-00001.safetensors").stat().st_size
    )
    tensors = load_file(str(output / "model-00001-of-00001.safetensors"))
    assert set(tensors) == {
        "mtp.layers.0.fc_embedding.weight",
        "mtp.shared.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert set(index["weight_map"]) == set(tensors)
    with pytest.raises(FileExistsError):
        exporter.export_checkpoint(source, output)
