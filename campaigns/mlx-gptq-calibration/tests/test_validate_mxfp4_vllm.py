from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_mxfp4_vllm.py"
CAMPAIGN = SCRIPT.parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json"
SPEC = importlib.util.spec_from_file_location("validate_mxfp4_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class ValidateMxfp4VllmTests(unittest.TestCase):
    def test_missing_required_package_fails_closed(self) -> None:
        error = validator.importlib.metadata.PackageNotFoundError("vllm")
        with patch.object(validator.importlib.metadata, "version", side_effect=error):
            with self.assertRaisesRegex(validator.VllmValidationError, "missing"):
                validator.package_versions()

    def test_campaign_pins_vllm_runtime(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text())
        self.assertEqual(campaign["validation"]["engine"], "vllm")
        self.assertEqual(campaign["validation"]["vllm"], "0.32.0")

    def test_accepts_weight_only_mxfp4_pack_quantized_config(self) -> None:
        config = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 32,
                        },
                        "input_activations": None,
                    }
                },
            }
        }
        self.assertEqual(
            validator.validate_quantization_config(config),
            config["quantization_config"],
        )

    def test_rejects_quantized_activations_or_wrong_format(self) -> None:
        base = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 32},
                        "input_activations": None,
                    }
                },
            },
        }
        quantized_activations = copy.deepcopy(base)
        quantized_activations["quantization_config"]["config_groups"]["group_0"][
            "input_activations"
        ] = {"num_bits": 8}
        quantized_activations["quantization_config"]["config_groups"]["group_0"][
            "output_activations"
        ] = {"num_bits": 8}
        with self.assertRaisesRegex(validator.VllmValidationError, "input or output activations"):
            validator.validate_quantization_config(quantized_activations)

        wrong_format = copy.deepcopy(base)
        wrong_format["quantization_config"]["format"] = "dense"
        with self.assertRaisesRegex(validator.VllmValidationError, "wrong compressed-tensors"):
            validator.validate_quantization_config(wrong_format)

    def test_rejects_malformed_groups_and_wrong_compressed_tensors_version(self) -> None:
        base = {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "version": "0.19.0",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 32},
                        "input_activations": None,
                        "output_activations": None,
                    }
                },
            }
        }
        validator.validate_quantization_config(base, "0.19.0")

        malformed = copy.deepcopy(base)
        malformed["quantization_config"]["config_groups"]["group_0"] = None
        with self.assertRaisesRegex(validator.VllmValidationError, "group entries"):
            validator.validate_quantization_config(malformed)

        wrong_version = copy.deepcopy(base)
        with self.assertRaisesRegex(validator.VllmValidationError, "version mismatch"):
            validator.validate_quantization_config(wrong_version, "0.18.0")

    def test_gpu_state_must_meet_campaign_minimum(self) -> None:
        gpu = {"name": "GPU", "capability": [8, 6]}
        self.assertEqual(validator.validate_gpu_state([gpu], 8), [gpu])
        with self.assertRaisesRegex(validator.VllmValidationError, "below campaign minimum"):
            validator.validate_gpu_state([gpu], 9)
        with self.assertRaisesRegex(validator.VllmValidationError, "unavailable"):
            validator.validate_gpu_state(None, 8)

    def test_artifact_identity_is_campaign_hash_bound(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text())
        output = campaign["output"]
        hashes = {
            "model.safetensors": output["text_model_safetensors_sha256"],
            "config.json": output["text_model_config_sha256"],
        }
        hashes.update(output["tokenizer_files"])
        with patch.object(validator, "sha256_file", side_effect=lambda path: hashes[path.name]):
            self.assertEqual(validator.verify_artifact_identity(Path("/model"), campaign), hashes)

        wrong = dict(hashes, **{"config.json": "0" * 64})
        with patch.object(validator, "sha256_file", side_effect=lambda path: wrong[path.name]):
            with self.assertRaisesRegex(validator.VllmValidationError, "artifact hashes"):
                validator.verify_artifact_identity(Path("/model"), campaign)

    def test_receipt_path_cannot_overwrite_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            validator.verify_receipt_path(model, Path(directory) / "receipt.json")
            expected = "inside the model directory"
            with self.assertRaisesRegex(validator.VllmValidationError, expected):
                validator.verify_receipt_path(model, model / "model.safetensors")


if __name__ == "__main__":
    unittest.main()
