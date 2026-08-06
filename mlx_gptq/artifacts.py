"""Calibration artifact store.

Layout of an artifact directory (produced by Stage A, consumed by Stage B):

    calib_dir/
      manifest.json            global settings + per-tensor metadata
      layer_0000.safetensors   {name}.q / {name}.scales / optional {name}.biases
      layer_0001.safetensors
      ...
      calib_tokens.npy         the exact token windows used (reproducibility)

Tensor names are RAW HF-checkpoint names (per-expert, e.g.
``model.layers.3.mlp.experts.17.gate_proj.weight``), which is what mlx-lm's
sanitize() consumes on the other side. 4-bit codes are nibble-packed
(two per byte, low nibble first) to halve artifact size.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file

_STR_TO_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.uint8 and q.shape[-1] % 2 == 0
    return q[..., 0::2] | (q[..., 1::2] << 4)


def unpack_nibbles(p: torch.Tensor) -> torch.Tensor:
    lo = p & 0xF
    hi = p >> 4
    out = torch.stack([lo, hi], dim=-1)
    return out.reshape(*p.shape[:-1], p.shape[-1] * 2)


class ArtifactWriter:
    def __init__(self, out_dir: str, header: dict):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.manifest_path = os.path.join(out_dir, "manifest.json")
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            for k, v in header.items():
                if k in ("mode", "bits", "group_size", "storage_dtype") and self.manifest.get(k) != v:
                    raise RuntimeError(
                        f"resume mismatch: manifest has {k}={self.manifest.get(k)}, run has {v}"
                    )
        else:
            self.manifest = dict(header)
            self.manifest["tensors"] = {}
            self.manifest["layers_done"] = []

    def layer_done(self, layer_idx: int) -> bool:
        return layer_idx in self.manifest["layers_done"]

    def layer_file(self, layer_idx: int) -> str:
        return os.path.join(self.dir, f"layer_{layer_idx:04d}.safetensors")

    def write_layer(self, layer_idx: int, tensors: dict, meta: dict):
        """tensors: name -> (q uint8 [out,in], scales, optional biases)."""
        flat = {}
        for name, (q, s, b) in tensors.items():
            bits = meta[name]["bits"]
            if bits == 4:
                q = pack_nibbles(q)
                meta[name]["q_packed"] = "nibble"
            else:
                meta[name]["q_packed"] = "raw"
            flat[f"{name}.q"] = q.contiguous()
            flat[f"{name}.scales"] = s.contiguous()
            if b is not None:
                flat[f"{name}.biases"] = b.contiguous()
                meta[name]["has_biases"] = True
            else:
                meta[name]["has_biases"] = False
            meta[name]["file"] = f"layer_{layer_idx:04d}.safetensors"
        save_file(flat, self.layer_file(layer_idx))
        self.manifest["tensors"].update(meta)
        self.manifest["layers_done"].append(layer_idx)
        self._flush()

    def _flush(self):
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.manifest, f)
        os.replace(tmp, self.manifest_path)


class ArtifactReader:
    def __init__(self, calib_dir: str):
        self.dir = calib_dir
        with open(os.path.join(calib_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.tensors = self.manifest["tensors"]
        self._cache_file = None
        self._cache = None

    @property
    def storage_dtype(self) -> torch.dtype:
        return _STR_TO_DTYPE[self.manifest["storage_dtype"]]

    def has(self, name: str) -> bool:
        return name in self.tensors

    def _load(self, fname: str) -> dict:
        if self._cache_file != fname:
            self._cache = load_file(os.path.join(self.dir, fname))
            self._cache_file = fname
        return self._cache

    def get(self, name: str):
        """Returns (q uint8 [out, in], scales, optional biases, info)."""
        info = self.tensors[name]
        blob = self._load(info["file"])
        q = blob[f"{name}.q"]
        if info["q_packed"] == "nibble":
            q = unpack_nibbles(q)
        biases = blob.get(f"{name}.biases")
        return q, blob[f"{name}.scales"], biases, info
