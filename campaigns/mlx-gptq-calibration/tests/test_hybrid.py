from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlx_gptq_calibration.hybrid import _read_header, apply_overlay


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def read_tensor(path: Path, name: str) -> bytes:
    header_size, header = _read_header(path)
    start, end = header[name]["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(8 + header_size + start)
        return handle.read(end - start)


class HybridTests(unittest.TestCase):
    def test_apply_overlay_replaces_only_matching_bf16_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            shard = model / "model.safetensors"
            router = "language_model.model.layers.0.router.proj.weight"
            other = "language_model.model.layers.0.self_attn.q_proj.weight"
            write_safetensors(
                shard,
                {
                    router: ("BF16", [2, 2], b"abcdefgh"),
                    other: ("BF16", [2, 2], b"12345678"),
                },
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {router: shard.name, other: shard.name},
                    }
                )
            )
            overlay = root / "overlay.safetensors"
            source = "model.language_model.layers.0.router.proj.weight"
            write_safetensors(overlay, {source: ("BF16", [2, 2], b"ABCDEFGH")})

            receipt = apply_overlay(
                model,
                overlay,
                "model.language_model.layers.",
                "language_model.model.layers.",
            )

            self.assertEqual(receipt["tensor_count"], 1)
            self.assertEqual(read_tensor(shard, router), b"ABCDEFGH")
            self.assertEqual(read_tensor(shard, other), b"12345678")
            self.assertNotEqual(
                receipt["tensors"][router]["before_sha256"],
                receipt["tensors"][router]["after_sha256"],
            )

    def test_apply_overlay_rejects_quantized_router_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            shard = model / "model.safetensors"
            target = "language_model.model.layers.0.router.proj.weight"
            write_safetensors(shard, {target: ("U32", [2, 1], b"abcdefgh")})
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {target: shard.name},
                    }
                )
            )
            overlay = root / "overlay.safetensors"
            source = "model.language_model.layers.0.router.proj.weight"
            write_safetensors(overlay, {source: ("BF16", [2, 2], b"ABCDEFGH")})

            with self.assertRaisesRegex(ValueError, "requires BF16"):
                apply_overlay(
                    model,
                    overlay,
                    "model.language_model.layers.",
                    "language_model.model.layers.",
                )


if __name__ == "__main__":
    unittest.main()
