from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlx_gptq_calibration.publication import (
    PublicationError,
    build_expected_files,
    require_basename,
    verify_artifact_files,
    verify_receipt_equivalence,
    verify_remote_artifact_files,
    verify_remote_files,
    verify_result_matches_packaging,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.artifact = root / "artifact"
        self.artifact.mkdir()
        files = {
            "model.safetensors": b"model-bytes",
            "config.json": b"config-bytes",
            "tokenizer.json": b"tokenizer-bytes",
        }
        for name, value in files.items():
            (self.artifact / name).write_bytes(value)
        self.packaging = {"output_files": {name: digest(value) for name, value in files.items()}}
        self.result = {
            "recommendation": {
                "artifact": {
                    "model_safetensors_sha256": digest(files["model.safetensors"]),
                    "config_sha256": digest(files["config.json"]),
                }
            }
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_verifies_artifact_result_and_expected_file_set(self) -> None:
        measured = verify_artifact_files(
            self.artifact,
            self.packaging,
            model_shard_sha256=digest(b"model-bytes"),
            config_sha256=digest(b"config-bytes"),
        )
        verify_result_matches_packaging(self.result, self.packaging)

        model_card = self.artifact.parent / "README.md"
        model_card.write_text("card\n")
        receipt = self.artifact.parent / "result.json"
        receipt.write_text("{}\n")
        expected = build_expected_files(
            self.artifact,
            measured,
            model_card,
            {"result.json": receipt},
        )
        self.assertEqual(
            set(expected),
            {
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "README.md",
                "receipts/qwen38-precision-ladder/result.json",
            },
        )

    def test_rejects_unexpected_artifact_file(self) -> None:
        (self.artifact / "extra").write_text("unexpected\n")
        with self.assertRaisesRegex(PublicationError, "unexpected=\\['extra'\\]"):
            verify_artifact_files(
                self.artifact,
                self.packaging,
                model_shard_sha256=digest(b"model-bytes"),
                config_sha256=digest(b"config-bytes"),
            )

    def test_rejects_result_and_packaging_disagreement(self) -> None:
        self.result["recommendation"]["artifact"]["config_sha256"] = "0" * 64
        with self.assertRaisesRegex(PublicationError, "config.json"):
            verify_result_matches_packaging(self.result, self.packaging)

    def test_verifies_matching_receipts_and_rejects_drift(self) -> None:
        local = self.artifact.parent / "local.json"
        volume = self.artifact.parent / "volume.json"
        local.write_text('{"value": 1}\n')
        volume.write_text('{"value": 1}\n')
        verify_receipt_equivalence({"result.json": local}, {"result.json": volume})
        volume.write_text('{"value": 2}\n')
        with self.assertRaisesRegex(PublicationError, "result.json"):
            verify_receipt_equivalence({"result.json": local}, {"result.json": volume})

    def test_rejects_receipt_name_set_drift(self) -> None:
        with self.assertRaisesRegex(PublicationError, "local and volume receipt names differ"):
            verify_receipt_equivalence(
                {"result.json": Path("local")},
                {"throughput.json": Path("volume")},
            )

    def test_remote_verification_ignores_directories_and_gitattributes(self) -> None:
        entries = {
            "receipts": SimpleNamespace(size=None),
            ".gitattributes": SimpleNamespace(size=10),
            "model.safetensors": SimpleNamespace(
                size=10,
                lfs=SimpleNamespace(sha256=digest(b"model-bytes")),
            ),
            "config.json": SimpleNamespace(size=12),
            "tokenizer.json": SimpleNamespace(size=14),
            "README.md": SimpleNamespace(size=5),
            "receipts/qwen38-precision-ladder/result.json": SimpleNamespace(size=3),
        }
        expected = {
            "model.safetensors": 10,
            "config.json": 12,
            "tokenizer.json": 14,
            "README.md": 5,
            "receipts/qwen38-precision-ladder/result.json": 3,
        }
        self.assertEqual(
            verify_remote_files(entries, expected),
            digest(b"model-bytes"),
        )
        del entries["README.md"]
        with self.assertRaisesRegex(PublicationError, "README.md"):
            verify_remote_files(entries, expected)

    def test_remote_artifact_subset_verification_allows_stale_evidence(self) -> None:
        entries = {
            "model.safetensors": SimpleNamespace(
                size=10,
                lfs=SimpleNamespace(sha256=digest(b"model-bytes")),
            ),
            "config.json": SimpleNamespace(size=12),
            "README.md": SimpleNamespace(size=1),
        }
        self.assertEqual(
            verify_remote_artifact_files(entries, {"model.safetensors": 10, "config.json": 12}),
            digest(b"model-bytes"),
        )

    def test_publication_names_must_be_basenames(self) -> None:
        for name in ("../escape", "nested/name", "C:\\escape", ".."):
            with self.subTest(name=name):
                with self.assertRaisesRegex(PublicationError, "invalid"):
                    require_basename(name, kind="test file")


if __name__ == "__main__":
    unittest.main()
