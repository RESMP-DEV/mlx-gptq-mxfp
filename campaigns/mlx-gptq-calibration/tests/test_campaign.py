from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from mlx_gptq_calibration.campaign import CampaignError, load_campaign
from mlx_gptq_calibration.cli import (
    DEFAULT_CAMPAIGN,
    inspect_local_model,
    remote_hfd_download_script,
    stage_a_argv,
)

GEMMA_CAMPAIGN = DEFAULT_CAMPAIGN.with_name("gemma4-26b-a4b-qat-mxfp4.json")
QWEN_MOE_CAMPAIGN = DEFAULT_CAMPAIGN.with_name("qwen3.6-35b-a3b-mxfp4.json")


class CampaignTests(unittest.TestCase):
    def test_default_campaign_is_valid_and_pinned(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        self.assertEqual(campaign.data["model"]["id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(campaign.data["dataset"]["id"], "RESMP-DEV/ptq-calibration-corpus")
        self.assertEqual(campaign.data["calibration"]["mode"], "mxfp4")
        self.assertEqual(campaign.data["calibration"]["group_size"], 32)

    def test_stage_a_command_uses_qwen_tokens_and_language_layers(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        argv = stage_a_argv(campaign, 4)
        self.assertIn("models/Qwen-Qwen3.8-27B-1d4bf0f", argv)
        self.assertIn("cuda:0,cuda:1,cuda:2,cuda:3", argv)
        self.assertIn("model.layers", argv)
        self.assertIn("model.language_model", argv)
        self.assertIn("language_model.model.layers", argv)
        self.assertIn("--calibration-tokens", argv)
        self.assertIn(
            ".local/share/mlx-gptq-calibration/qwen3.8-27b-mxfp4/inputs/calibration.npy",
            argv,
        )
        self.assertNotIn("--dataset", argv)

    def test_stage_a_command_preserves_heterogeneous_gpu_budgets(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        argv = stage_a_argv(campaign, [0, 2, 3], [4.2, 25.8, 83.4])
        self.assertIn("cuda:0,cuda:2,cuda:3", argv)
        self.assertIn("4.2,25.8,83.4", argv)

    def test_remote_download_uses_hfd_without_xet(self) -> None:
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        script = remote_hfd_download_script(campaign)
        self.assertIn('"$hfd_bin" download Qwen/Qwen3.8-27B', script)
        self.assertIn("--verify full", script)
        self.assertIn('"$hfd_bin" verify Qwen/Qwen3.8-27B', script)
        self.assertIn("HF_HUB_DISABLE_XET=1", script)
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", script)
        self.assertNotIn("hf download", script)
        self.assertIn("--output models/Qwen-Qwen3.8-27B-1d4bf0f", script)

    def test_shared_remote_model_directory_can_be_overridden(self) -> None:
        data = json.loads(DEFAULT_CAMPAIGN.read_text())
        data["compute"]["remote_model_dir"] = "models/shared-qwen"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps(data))
            self.assertEqual(load_campaign(path).remote_model_dir, "models/shared-qwen")

    def test_gemma_campaign_targets_the_moe_not_the_drafter(self) -> None:
        campaign = load_campaign(GEMMA_CAMPAIGN)
        self.assertEqual(
            campaign.data["model"]["id"],
            "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        )
        self.assertEqual(campaign.data["model"]["model_type"], "gemma4")
        self.assertEqual(
            campaign.data["model"]["assistant"]["id"],
            "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
        )
        argv = stage_a_argv(campaign, 4)
        self.assertIn("model.language_model.layers", argv)
        self.assertIn("language_model.model.layers", argv)
        self.assertNotIn("--checkpoint-model-prefix", argv)

    def test_gemma_hybrid_is_pinned_and_router_is_kept_bf16(self) -> None:
        campaign = load_campaign(GEMMA_CAMPAIGN)
        hybrid = campaign.data["hybrid"]
        self.assertEqual(hybrid["source"]["id"], "google/gemma-4-26B-A4B-it")
        self.assertEqual(
            hybrid["source"]["revision"],
            "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        )
        keep = re.compile(campaign.data["packing"]["keep_bf16_regex"])
        self.assertIsNotNone(keep.search("language_model.model.layers.0.router.proj"))

    def test_qwen36_moe_control_is_pinned_and_keeps_both_gates_bf16(self) -> None:
        campaign = load_campaign(QWEN_MOE_CAMPAIGN)
        self.assertEqual(campaign.data["model"]["id"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(campaign.data["model"]["model_type"], "qwen3_5_moe")
        keep = re.compile(campaign.data["packing"]["keep_bf16_regex"])
        self.assertIsNotNone(keep.search("language_model.model.layers.0.mlp.gate"))
        self.assertIsNotNone(keep.search("language_model.model.layers.0.mlp.shared_expert_gate"))

    def test_local_model_requires_all_indexed_shards(self) -> None:
        campaign = load_campaign(GEMMA_CAMPAIGN)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 51611872412},
                        "weight_map": {"a": "model-00001-of-00002.safetensors"},
                    }
                )
            )
            with self.assertRaisesRegex(CampaignError, "missing 1 shard"):
                inspect_local_model(campaign, root)
            header = json.dumps(
                {"weight": {"dtype": "U8", "shape": [7], "data_offsets": [0, 7]}}
            ).encode()
            (root / "model-00001-of-00002.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"fixture"
            )
            receipt = inspect_local_model(campaign, root)
            self.assertEqual(receipt["weight_shards"], 1)

    def test_local_model_rejects_partial_safetensors_file(self) -> None:
        campaign = load_campaign(GEMMA_CAMPAIGN)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 51611872412},
                        "weight_map": {"a": "model.safetensors"},
                    }
                )
            )
            header = json.dumps(
                {"weight": {"dtype": "U8", "shape": [100], "data_offsets": [0, 100]}}
            ).encode()
            (root / "model.safetensors").write_bytes(
                len(header).to_bytes(8, "little") + header + b"partial"
            )
            with self.assertRaisesRegex(CampaignError, "truncated shard"):
                inspect_local_model(campaign, root)

    def test_rejects_invalid_native_mxfp_shape(self) -> None:
        data = json.loads(DEFAULT_CAMPAIGN.read_text())
        data["calibration"]["group_size"] = 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(CampaignError, "mxfp4 requires"):
                load_campaign(path)


if __name__ == "__main__":
    unittest.main()
