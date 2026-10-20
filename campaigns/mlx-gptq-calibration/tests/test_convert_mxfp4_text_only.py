from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_mxfp4_text_only.py"
SPEC = importlib.util.spec_from_file_location("convert_mxfp4_text_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConvertMxfp4TextOnlyTests(unittest.TestCase):
    def test_flattens_conditional_text_config(self) -> None:
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
            },
            "quantization_config": {
                "format": "mxfp4-pack-quantized",
                "ignore": ["lm_head", "re:^mtp.*"],
            },
        }
        flattened = converter.flatten_config(config, "Qwen3_5ForCausalLM", "qwen3_5_text")
        self.assertEqual(flattened["architectures"], ["Qwen3_5ForCausalLM"])
        self.assertEqual(flattened["model_type"], "qwen3_5_text")
        self.assertEqual(flattened["hidden_size"], 5120)
        self.assertEqual(flattened["quantization_config"]["ignore"], ["lm_head"])

    def test_missing_quantization_config_fails_closed(self) -> None:
        config = {
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
            }
        }
        with self.assertRaisesRegex(converter.TextOnlyPackagingError, "quantization_config"):
            converter.flatten_config(config, "Qwen3_5ForCausalLM", "qwen3_5_text")

    def test_rewrites_language_keys_and_drops_auxiliary_weights(self) -> None:
        checkpoint_prefix = "model.language_model"
        artifact_prefix = "model."
        self.assertEqual(
            converter.tensor_key(
                "model.language_model.layers.0.mlp.down_proj.weight_packed",
                checkpoint_prefix,
                artifact_prefix,
            ),
            "model.layers.0.mlp.down_proj.weight_packed",
        )
        self.assertEqual(
            converter.tensor_key("lm_head.weight", checkpoint_prefix, artifact_prefix),
            "lm_head.weight",
        )
        self.assertIsNone(converter.tensor_key("mtp.fc.weight", checkpoint_prefix, artifact_prefix))
        self.assertIsNone(
            converter.tensor_key(
                "model.visual.blocks.0.mlp.linear_fc1.weight",
                checkpoint_prefix,
                artifact_prefix,
            )
        )

    def test_source_index_must_have_main_and_mtp_shards(self) -> None:
        with self.assertRaisesRegex(converter.TextOnlyPackagingError, "main and MTP"):
            converter.verify_source_index(
                {"weight_map": {"a": "model.safetensors"}},
                set(),
                "model.language_model",
                "model.",
            )

    def test_source_index_rejects_text_tensor_in_unread_shard(self) -> None:
        index = {
            "weight_map": {
                "lm_head.weight": "model.safetensors",
                "model.language_model.embed_tokens.weight": "model_mtp.safetensors",
                "mtp.fc.weight": "model_mtp.safetensors",
            }
        }
        loaded = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
        with self.assertRaisesRegex(converter.TextOnlyPackagingError, "wrong_shard"):
            converter.verify_source_index(index, loaded, "model.language_model", "model.")

    def test_source_index_accepts_complete_main_shard_text_tensors(self) -> None:
        index = {
            "weight_map": {
                "model.language_model.embed_tokens.weight": "model.safetensors",
                "mtp.fc.weight": "model_mtp.safetensors",
            }
        }
        loaded = {"model.language_model.embed_tokens.weight"}
        converter.verify_source_index(index, loaded, "model.language_model", "model.")

    def test_restores_pinned_tokenizer_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()
            for name in converter.TOKENIZER_FILE_NAMES:
                (tokenizer_source / name).write_bytes(f"pinned-{name}".encode())
            expected = {
                name: hashlib.sha256(f"pinned-{name}".encode()).hexdigest()
                for name in converter.TOKENIZER_FILE_NAMES
            }

            hashes = converter.copy_pinned_tokenizer(tokenizer_source, output, expected)

            self.assertEqual(set(hashes), set(converter.TOKENIZER_FILE_NAMES))
            for name in converter.TOKENIZER_FILE_NAMES:
                self.assertEqual((output / name).read_bytes(), f"pinned-{name}".encode())

    def test_missing_pinned_tokenizer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()

            expected = "pinned tokenizer source is missing"
            complete = {name: "0" * 64 for name in converter.TOKENIZER_FILE_NAMES}
            with self.assertRaisesRegex(converter.TextOnlyPackagingError, expected):
                converter.verify_pinned_tokenizer(tokenizer_source, complete)

    def test_wrong_pinned_tokenizer_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()
            expected = {}
            for name in converter.TOKENIZER_FILE_NAMES:
                content = f"pinned-{name}".encode()
                (tokenizer_source / name).write_bytes(content)
                expected[name] = hashlib.sha256(content).hexdigest()
            expected["tokenizer.json"] = "0" * 64

            with self.assertRaisesRegex(converter.TextOnlyPackagingError, "tokenizer source hash"):
                converter.verify_pinned_tokenizer(tokenizer_source, expected)

    def test_copy_pinned_tokenizer_does_not_reverify_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_source = root / "source-checkpoint"
            output = root / "artifact"
            tokenizer_source.mkdir()
            output.mkdir()
            verified = {}
            for name in converter.TOKENIZER_FILE_NAMES:
                (tokenizer_source / name).write_bytes(f"pinned-{name}".encode())
                verified[name] = hashlib.sha256(f"pinned-{name}".encode()).hexdigest()

            returned = converter.copy_pinned_tokenizer(tokenizer_source, output, verified)

            self.assertEqual(returned, verified)

    def test_staged_tokenizer_hashes_are_checked_before_release(self) -> None:
        hashes = {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in converter.TOKENIZER_FILE_NAMES
        }
        converter.verify_staged_tokenizer_hashes(hashes, hashes)
        wrong = dict(hashes)
        wrong["tokenizer.json"] = "0" * 64
        with self.assertRaisesRegex(converter.TextOnlyPackagingError, "staged tokenizer"):
            converter.verify_staged_tokenizer_hashes(wrong, hashes)


if __name__ == "__main__":
    unittest.main()
