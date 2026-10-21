from __future__ import annotations

from mlx_gptq.flashnext_mtp_eval import paired_test


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
