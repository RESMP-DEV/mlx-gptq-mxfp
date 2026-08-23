import pytest

from mlx_gptq.calibrate import parse_vram_budgets


def test_single_budget_is_broadcast_to_all_devices():
    assert parse_vram_budgets("12", ["cuda:0", "cuda:1"]) == {
        "cuda:0": 12.0,
        "cuda:1": 12.0,
    }


def test_per_device_budgets_preserve_heterogeneous_capacity():
    assert parse_vram_budgets("4.2,18.6,83.4", ["cuda:0", "cuda:1", "cuda:2"]) == {
        "cuda:0": 4.2,
        "cuda:1": 18.6,
        "cuda:2": 83.4,
    }


def test_per_device_budget_count_must_match():
    with pytest.raises(ValueError, match="one comma-separated value per device"):
        parse_vram_budgets("4,8", ["cuda:0", "cuda:1", "cuda:2"])
