from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mlx_gptq.flashnext_mtp_eval import load_ggml_candidate, paired_test
from mlx_gptq.ggml_native import GGML_FORMATS, NativeGGMLQuantizer
from safetensors import safe_open
from safetensors.torch import save_file


def _record(ce: float, top1: float) -> dict[str, float]:
    return {
        "index": 0.0,
        "source_position": 0.0,
        "loss_ce": ce,
        "loss_tv": 0.0,
        "top1_match": top1,
    }


def test_paired_test_reports_discordant_counts_and_significance() -> None:
    base = [_record(2.0, 0.0)] * 10 + [_record(2.0, 1.0)] * 5
    candidate = [_record(1.0, 1.0)] * 10 + [_record(2.0, 1.0)] * 5
    result = paired_test(base, candidate)
    assert result["pairs"] == 15
    assert result["top1_candidate_wins"] == 10
    assert result["top1_candidate_losses"] == 0
    assert result["mcnemar_exact_p_value"] < 0.01
    assert result["ce_mean_difference"] < 0.0
    assert result["ce_difference_ci95_excludes_zero"] is True


def test_paired_test_reports_no_effect_when_nothing_changes() -> None:
    records = [_record(1.5, 1.0)] * 8
    result = paired_test(records, list(records))
    assert result["top1_candidate_wins"] == 0
    assert result["top1_candidate_losses"] == 0
    assert result["mcnemar_exact_p_value"] == 1.0
    assert result["ce_mean_difference"] == 0.0
    assert result["ce_difference_ci95_excludes_zero"] is False


def test_paired_test_rejects_mismatched_lengths() -> None:
    try:
        paired_test([_record(1.0, 1.0)], [])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected a ValueError for mismatched lengths")


def test_safetensors_reader_preserves_candidate_format_metadata(tmp_path) -> None:
    path = tmp_path / "candidate.safetensors"
    save_file(
        {"layer.blocks": torch.zeros(1, 1, 18, dtype=torch.uint8)},
        str(path),
        metadata={"module_formats": '{"layer":"q4_0"}'},
    )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert handle.metadata()["module_formats"] == '{"layer":"q4_0"}'


def test_load_ggml_candidate_applies_packed_blocks(tmp_path) -> None:
    library = Path(
        "/Users/kearm/Strata/build/llama-tools/bin/libggml-base.0.24.0.dylib"
    )
    if not library.is_file():
        pytest.skip("local native ggml-base library is unavailable")

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(256, 256, bias=False)

    model = Toy()
    original = model.layer.weight.detach().clone()
    quantizer = NativeGGMLQuantizer(str(library))
    blocks, _ = quantizer.quantize_blocks(model.layer.weight, GGML_FORMATS["iq4_xs"])
    path = tmp_path / "candidate.safetensors"
    save_file(
        {"layer.blocks": blocks},
        str(path),
        metadata={
            "requested_format": "iq4_xs",
            "module_formats": '{"layer":"iq4_xs"}',
        },
    )
    result = load_ggml_candidate(model, path, library)
    assert result["candidate_source"] == "native_ggml"
    assert result["applied_modules"] == 1
    assert not torch.equal(model.layer.weight.detach(), original)
