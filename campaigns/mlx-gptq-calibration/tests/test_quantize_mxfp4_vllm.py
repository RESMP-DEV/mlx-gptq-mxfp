from __future__ import annotations

import array
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlx_gptq_calibration.campaign import load_campaign

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/quantize_mxfp4_vllm.py"
SPEC = importlib.util.spec_from_file_location("quantize_mxfp4_vllm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)

CAMPAIGN = SCRIPT.parents[1] / "campaigns/qwen3.8-27b-mxfp4-vllm.json"


class QuantizeMxfp4VllmTests(unittest.TestCase):
    def test_campaign_is_pinned_for_compressed_tensors(self) -> None:
        campaign = load_campaign(CAMPAIGN)
        self.assertEqual(campaign.data["output"]["format"], "compressed-tensors")
        self.assertEqual(campaign.data["output"]["scheme"], "MXFP4A16")
        self.assertEqual(
            campaign.data["engine"]["commit"], "6e4b96e467c243cb988e87c0fe97b4d51359f983"
        )
        self.assertEqual(campaign.data["runtime"]["llmcompressor"], "0.14.0")
        self.assertNotEqual(campaign.data["model"].get("pipeline_content_manifest_sha256"), None)
        self.assertEqual(
            campaign.data["engine"]["commit_scope"],
            "recipe_reference_not_installed_build_provenance",
        )
        self.assertEqual(campaign.data["validation"]["vllm"], "0.32.0")
        self.assertEqual(campaign.data["validation"]["ampere_behavior"], "Marlin W4A16")

    def test_runtime_mismatch_fails_closed(self) -> None:
        data = json.loads(CAMPAIGN.read_text())
        versions = {
            name: value
            for name, value in data["runtime"].items()
            if name != "llmcompressor_record_sha256"
        }
        record_hash = data["runtime"]["llmcompressor_record_sha256"]
        with (
            patch.object(quantize, "package_versions", return_value=versions),
            patch.object(quantize, "llmcompressor_record_sha256", return_value=record_hash),
        ):
            verified = quantize.verify_runtime(data)
        self.assertEqual(verified["llmcompressor"], "0.14.0")
        self.assertEqual(verified["llmcompressor_record_sha256"], record_hash)

        versions["transformers"] = "0.0.0"
        with (
            patch.object(quantize, "package_versions", return_value=versions),
            patch.object(quantize, "llmcompressor_record_sha256", return_value=record_hash),
        ):
            with self.assertRaisesRegex(quantize.QuantizationContractError, "runtime package"):
                quantize.verify_runtime(data)

        versions["transformers"] = data["runtime"]["transformers"]
        with patch.object(quantize, "package_versions", return_value=versions):
            with patch.object(quantize, "llmcompressor_record_sha256", return_value="0" * 64):
                with self.assertRaisesRegex(quantize.QuantizationContractError, "RECORD hash"):
                    quantize.verify_runtime(data)

    def test_missing_engine_section_fails_closed(self) -> None:
        data = json.loads(CAMPAIGN.read_text())
        del data["engine"]
        versions = {
            name: value
            for name, value in data["runtime"].items()
            if name != "llmcompressor_record_sha256"
        }
        with patch.object(quantize, "package_versions", return_value=versions):
            with self.assertRaisesRegex(quantize.QuantizationContractError, "engine section"):
                quantize.verify_runtime(data)

    def test_missing_required_package_fails_closed(self) -> None:
        error = quantize.importlib.metadata.PackageNotFoundError("llmcompressor")
        with patch.object(quantize.importlib.metadata, "version", side_effect=error):
            with self.assertRaisesRegex(quantize.QuantizationContractError, "missing"):
                quantize.package_versions()

    def test_calibration_hash_and_shape_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npy"
            base = b"{'descr':'<i8','fortran_order':False,'shape':(2,4),}"
            header = base + b" " * (64 - 10 - len(base) - 1) + b"\n"
            path.write_bytes(
                b"\x93NUMPY\x01\x00"
                + len(header).to_bytes(2, "little")
                + header
                + array.array("q", range(8)).tobytes()
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fake_numpy = SimpleNamespace(
                load=lambda source, mmap_mode: SimpleNamespace(
                    shape=(2, 4),
                    min=lambda: 0,
                    tolist=lambda: [[0, 1, 2, 3], [4, 5, 6, 7]],
                ),
                isnan=lambda matrix: SimpleNamespace(any=lambda: False),
            )
            with patch.dict(sys.modules, {"numpy": fake_numpy}):
                loaded = quantize.load_calibration(path, digest, (2, 4))
            self.assertEqual(loaded.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
            with self.assertRaisesRegex(quantize.QuantizationContractError, "shape mismatch"):
                wrong_shape = SimpleNamespace(
                    load=lambda source, mmap_mode: SimpleNamespace(shape=(4, 2))
                )
                with patch.dict(sys.modules, {"numpy": wrong_shape}):
                    quantize.load_calibration(path, digest, (2, 4))

    def test_nan_calibration_tokens_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npy"
            path.write_bytes(b"calibration")
            digest = hashlib.sha256(b"calibration").hexdigest()
            matrix = SimpleNamespace(shape=(2, 4), min=lambda: 0)
            fake_numpy = SimpleNamespace(
                load=lambda source, mmap_mode: matrix,
                isnan=lambda value: SimpleNamespace(any=lambda: True),
            )
            with patch.dict(sys.modules, {"numpy": fake_numpy}):
                with self.assertRaisesRegex(quantize.QuantizationContractError, "NaN"):
                    quantize.load_calibration(path, digest, (2, 4))

    def test_source_receipt_repository_and_both_revisions_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "source-model"
            campaign = json.loads(CAMPAIGN.read_text())
            revision = campaign["model"]["revision"]
            receipt = {
                "status": "verified",
                "repo": campaign["model"]["id"],
                "requested_revision": revision,
                "resolved_revision": revision,
                "output": str(model_path),
                "failures": [],
            }
            self.assertEqual(quantize.verify_source_receipt(receipt, campaign, model_path), receipt)

            receipt["resolved_revision"] = "0" * 40
            expected = "source model verification"
            with self.assertRaisesRegex(quantize.QuantizationContractError, expected):
                quantize.verify_source_receipt(receipt, campaign, model_path)

            receipt["resolved_revision"] = revision
            receipt["output"] = str(model_path)
            receipt["failures"] = ["checksum mismatch"]
            with self.assertRaisesRegex(quantize.QuantizationContractError, expected):
                quantize.verify_source_receipt(receipt, campaign, model_path)

            receipt["resolved_revision"] = revision
            receipt["failures"] = []
            receipt["output"] = str(Path(directory) / "different-model")
            expected = "source model verification"
            with self.assertRaisesRegex(quantize.QuantizationContractError, expected):
                quantize.verify_source_receipt(receipt, campaign, model_path)

    def test_receipt_parent_is_created_and_made_writable_before_long_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "new-parent" / "receipt.json"
            quantize.prepare_receipt_path(receipt)
            self.assertTrue(receipt.parent.is_dir())
            self.assertFalse(receipt.exists())

    def test_output_and_receipt_paths_are_prepared_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            receipt = root / "receipts" / "receipt.json"
            quantize.prepare_output_and_receipt_paths(output, receipt)
            self.assertTrue(output.is_dir())
            self.assertFalse(receipt.exists())

    def test_existing_output_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.write_text("not a directory")
            receipt = root / "receipt.json"
            with self.assertRaisesRegex(quantize.QuantizationContractError, "not a directory"):
                quantize.prepare_output_and_receipt_paths(output, receipt)

    def test_quantization_gpu_preflight_enforces_campaign_minimums(self) -> None:
        campaign = json.loads(CAMPAIGN.read_text())
        gpu = {
            "name": "NVIDIA GeForce RTX 3090 Ti",
            "capability": [8, 6],
            "total_bytes": campaign["compute"]["minimum_vram_mib"] * 1024 * 1024,
        }
        with patch.object(quantize, "gpu_state", return_value=[gpu]):
            self.assertEqual(quantize.validate_quantization_gpus(campaign), [gpu])

        insufficient = dict(gpu, total_bytes=1024)
        with patch.object(quantize, "gpu_state", return_value=[insufficient]):
            with self.assertRaisesRegex(quantize.QuantizationContractError, "insufficient VRAM"):
                quantize.validate_quantization_gpus(campaign)

        old_gpu = dict(gpu, capability=[7, 5])
        with patch.object(quantize, "gpu_state", return_value=[old_gpu]):
            with self.assertRaisesRegex(
                quantize.QuantizationContractError, "minimum compute capability"
            ):
                quantize.validate_quantization_gpus(campaign)

    def test_receipt_inside_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            receipt = output / "receipt.json"
            with self.assertRaisesRegex(quantize.QuantizationContractError, "inside"):
                quantize.prepare_output_and_receipt_paths(output, receipt)

    def test_populated_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "existing.txt").write_text("do not overwrite")
            receipt = root / "receipt.json"
            with self.assertRaisesRegex(quantize.QuantizationContractError, "not empty"):
                quantize.prepare_output_and_receipt_paths(output, receipt)

    def test_checkpoint_manifest_hashes_files_size_and_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "b.safetensors").write_bytes(b"second")
            nested = model / "nested"
            nested.mkdir()
            (nested / "a.json").write_bytes(b"first")

            manifest = quantize.checkpoint_manifest_sha256(model)

            self.assertEqual(manifest, quantize.checkpoint_manifest_sha256(model))
            (nested / "a.json").write_bytes(b"changed")
            self.assertNotEqual(manifest, quantize.checkpoint_manifest_sha256(model))


if __name__ == "__main__":
    unittest.main()
