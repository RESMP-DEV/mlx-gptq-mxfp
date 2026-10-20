from __future__ import annotations

import unittest

from mlx_gptq_calibration.campaign import CampaignError
from mlx_gptq_calibration.cli import _require_satisfied, solver_vram_gb


class ProbeTests(unittest.TestCase):
    def test_rejects_host_without_cuda(self) -> None:
        probe = {
            "gpus": [],
            "requirements": {
                "minimum_vram_mib": 8000,
                "eligible_gpu_indices": [],
                "satisfied": False,
            },
        }
        with self.assertRaisesRegex(CampaignError, "at least 8000 MiB"):
            _require_satisfied(probe, allow=False)

    def test_override_is_explicit(self) -> None:
        probe = {
            "gpus": [],
            "requirements": {
                "minimum_vram_mib": 8000,
                "eligible_gpu_indices": [],
                "satisfied": False,
            },
        }
        _require_satisfied(probe, allow=True)

    def test_solver_budget_reserves_room_for_streamed_layer(self) -> None:
        from mlx_gptq_calibration.campaign import load_campaign
        from mlx_gptq_calibration.cli import DEFAULT_CAMPAIGN

        campaign = load_campaign(DEFAULT_CAMPAIGN)
        budget = solver_vram_gb(campaign, [{"memory_mib": 8151, "memory_free_mib": 8151}])
        self.assertEqual(budget, [4.2])

    def test_solver_budget_uses_live_free_memory(self) -> None:
        from mlx_gptq_calibration.campaign import load_campaign
        from mlx_gptq_calibration.cli import DEFAULT_CAMPAIGN

        campaign = load_campaign(DEFAULT_CAMPAIGN)
        budget = solver_vram_gb(campaign, [{"memory_mib": 24576, "memory_free_mib": 6144}])
        self.assertEqual(budget, [3.0])

    def test_solver_budget_preserves_each_gpu_capacity(self) -> None:
        from mlx_gptq_calibration.campaign import load_campaign
        from mlx_gptq_calibration.cli import DEFAULT_CAMPAIGN

        campaign = load_campaign(DEFAULT_CAMPAIGN)
        budget = solver_vram_gb(
            campaign,
            [
                {"memory_mib": 24576, "memory_free_mib": 24000},
                {"memory_mib": 32768, "memory_free_mib": 32000},
                {"memory_mib": 98304, "memory_free_mib": 96000},
            ],
        )
        self.assertEqual(budget, [18.6, 25.8, 83.4])


if __name__ == "__main__":
    unittest.main()
