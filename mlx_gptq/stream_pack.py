"""Streaming Stage B packer: HF BF16 source + Stage A GPTQ artifacts -> MLX model.

The dense `pack.py` path runs `mlx_lm.convert` on the whole model, which needs
the full BF16 checkpoint resident. FlashNext is 335 GiB, so that cannot run on a
128 GiB host. This module assembles the MLX checkpoint one unit at a time:

  1. Read the HF index and open source shards lazily.
  2. For each unit (one decoder layer, or the top-level tensors):
     load only that unit's source tensors, apply the MLX naming transforms,
     inject the Stage A GPTQ planes where present, RTN the rest, write a shard.
  3. Free the unit and continue.

Peak host memory is one unit of BF16 weights plus its quantized output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import ArtifactReader
from .grid import SUPPORTED_MODES, mode_defaults, rtn_quantize_mxfp
from .packing import pack_q

log = logging.getLogger("mlx_gptq")

NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.")
PER_EXPERT_RE = re.compile(
    r"(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$"
)

FOLD_ONE = (
    "q_layernorm.weight",
    "k_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "hc_norm.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
)


def sanitize_name(k: str, fold: bool) -> str | None:
    """HF checkpoint key -> MLX module-tree key. None means drop."""
    if k.startswith(("mtp.", "model.mtp.")):
        return None
    if k.startswith(("model.visual.", "visual.", "vision_tower.")):
        return None
    if k.startswith("model.language_model."):
        k = "model." + k[len("model.language_model.") :]
    elif k.startswith("language_model."):
        k = k[len("language_model.") :]
    k = NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1.", k)
    return k


def fold_applies(key: str) -> bool:
    return key.endswith(FOLD_ONE)


def unit_of(mlx_key: str) -> str:
    m = re.match(r"^model\.layers\.(\d+)\.", mlx_key)
    if m:
        return f"layer.{int(m.group(1)):04d}"
    return "top"


class StreamingPacker:
    def __init__(
        self,
        hf_path: str,
        calib: str,
        out_path: str,
        mode: str,
        bits: int,
        group_size: int,
        exclude: str | None,
        storage_dtype: str = "bfloat16",
    ):
        self.hf = Path(hf_path)
        self.out = Path(out_path)
        self.mode = mode
        self.bits = bits
        self.group_size = group_size
        self.exclude = re.compile(exclude) if exclude else None
        self.storage_dtype = np.dtype(
            np.float32 if storage_dtype == "float32" else np.float16
        )
        self.scale_dtype = np.uint8 if mode != "affine" else self.storage_dtype
        self.torch_dtype = (
            torch.float32 if storage_dtype == "float32" else torch.bfloat16
        )
        self.reader = ArtifactReader(calib)
        index_path = self.hf / "model.safetensors.index.json"
        if index_path.exists():
            self.index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = self.index["weight_map"]
        else:
            # Single-file checkpoint: synthesize the map from the file itself.
            from safetensors import safe_open as _so

            single = next(self.hf.glob("*.safetensors"))
            with _so(str(single), framework="pt") as f:
                self.weight_map = {k: single.name for k in list(f.keys())}
            self.index = {"weight_map": self.weight_map}
        self._open: dict[str, object] = OrderedDict()
        self._open_cap = 8
        self.stats = defaultdict(int)
        self.used_artifacts: set[str] = set()

    # ---------------------------------------------------------------- source
    def _reader_for(self, fname: str):
        fh = self._open.get(fname)
        if fh is None:
            fh = safe_open(str(self.hf / fname), framework="pt", device="cpu")
            self._open[fname] = fh
            while len(self._open) > self._open_cap:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(fname)
        return fh

    def source_files_for(self, keys) -> set[str]:
        return {self.weight_map[k] for k in keys if k in self.weight_map}

    # ------------------------------------------------------------- quantize
    def _quantizable(self, key: str, arr: np.ndarray) -> bool:
        if not key.endswith(".weight"):
            return False
        if arr.ndim not in (2, 3):
            return False
        if arr.shape[-1] % self.group_size:
            return False
        module_key = key.removesuffix(".weight")
        if module_key == "lm_head" or module_key.endswith(".lm_head"):
            return False
        if module_key.endswith("embed_tokens"):
            return False
        if self.exclude and self.exclude.search(module_key):
            return False
        return "ngram_embedding.shards." not in key

    def emit(self, mlx_key: str, arr: np.ndarray) -> dict[str, np.ndarray]:
        base = mlx_key[: -len(".weight")]
        if arr.ndim == 3 and ".switch_mlp." in mlx_key:
            experts = arr.shape[0]
            artifacts = [
                mlx_key.replace(".switch_mlp.", f".experts.{expert}.")
                for expert in range(experts)
            ]
            if all(self.reader.has(artifact) for artifact in artifacts):
                qs, scale_sets = [], []
                for artifact in artifacts:
                    q, scales, _biases, info = self.reader.get(artifact)
                    if (
                        info.get("bits") != self.bits
                        or info.get("group_size") != self.group_size
                    ):
                        raise RuntimeError(
                            f"{artifact}: artifact {info['bits']}b/"
                            f"g{info['group_size']} != config "
                            f"{self.bits}b/g{self.group_size}"
                        )
                    qs.append(q.numpy())
                    scale_sets.append(scales.numpy())
                    self.used_artifacts.add(artifact)
                words = pack_q(np.stack(qs), self.bits)
                scales = np.stack(scale_sets)
                self.stats["injected"] += experts
                return {
                    f"{base}.weight": words,
                    f"{base}.scales": scales.astype(self.scale_dtype),
                }
        if self.reader.has(mlx_key) and self._quantizable(mlx_key, arr):
            q, s, b, info = self.reader.get(mlx_key)
            if (
                info.get("bits") != self.bits
                or info.get("group_size") != self.group_size
            ):
                raise RuntimeError(
                    f"{mlx_key}: artifact {info['bits']}b/g{info['group_size']} "
                    f"!= config {self.bits}b/g{self.group_size}"
                )
            words = pack_q(q.numpy(), self.bits)
            out = {
                f"{base}.weight": words,
                f"{base}.scales": np.asarray(s.numpy()).astype(self.scale_dtype),
            }
            if b is not None:
                out[f"{base}.biases"] = np.asarray(b.float().numpy()).astype(
                    self.storage_dtype
                )
            self.stats["injected"] += 1
            self.used_artifacts.add(mlx_key)
            return out
        if self._quantizable(mlx_key, arr):
            import torch

            w = arr.contiguous().to(torch.float32)
            q, s, _b, _dq = rtn_quantize_mxfp(w.unsqueeze(0), self.mode)
            words = pack_q(q[0].numpy(), self.bits)
            out = {
                f"{base}.weight": words,
                f"{base}.scales": np.asarray(s[0].numpy()).astype(self.scale_dtype),
            }
            self.stats["rtn"] += 1
            return out
        self.stats["kept"] += 1
        return {mlx_key: arr}

    # ----------------------------------------------------------------- units
    def plan(self) -> dict[str, list[str]]:
        units: dict[str, list[str]] = defaultdict(list)
        for k in self.weight_map:
            sk = sanitize_name(k, fold=False)
            if sk is None:
                continue
            m = re.search(r"\.ngram_embedding\.shards\.(\d+)\.", sk)
            if m:
                units[f"ple.shard.{int(m.group(1)):04d}"].append(k)
                continue
            if sk.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                units[unit_of(sk)].append(k)
                continue
            units[unit_of(sk)].append(k)
        return units

    def pack_unit(self, unit: str, hf_keys: list[str]) -> dict[str, np.ndarray]:
        tensors: dict[str, np.ndarray] = {}
        for hf_key in hf_keys:
            fname = self.weight_map[hf_key]
            t = self._reader_for(fname).get_tensor(hf_key)
            # Keep source tensors in their original dtype, normally bfloat16.
            tensors[hf_key] = (
                t.contiguous().to(self.torch_dtype)
                if t.is_floating_point()
                else t.contiguous()
            )
            del t

        # group per-expert tensors for stacking
        per_expert: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        out: dict[str, np.ndarray] = {}
        fold = any(
            k.startswith(("model.language_model.", "language_model.")) for k in tensors
        )

        for hf_key, arr in tensors.items():
            k = sanitize_name(hf_key, fold)
            if k is None:
                continue
            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                mid = arr.shape[-2] // 2
                out.update(
                    self.emit(base + "switch_mlp.gate_proj.weight", arr[..., :mid, :])
                )
                out.update(
                    self.emit(base + "switch_mlp.up_proj.weight", arr[..., mid:, :])
                )
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")]
                out.update(self.emit(base + "switch_mlp.down_proj.weight", arr))
                continue
            m = PER_EXPERT_RE.match(k)
            if m:
                base, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                if proj == "gate_up_proj":
                    mid = arr.shape[-2] // 2
                    per_expert[base][idx]["gate_proj"] = arr[..., :mid, :]
                    per_expert[base][idx]["up_proj"] = arr[..., mid:, :]
                else:
                    per_expert[base][idx][proj] = arr
                continue
            if k.endswith("conv1d.weight") and arr.ndim == 3 and arr.shape[1] == 1:
                arr = (
                    arr.permute(0, 2, 1).contiguous()
                    if isinstance(arr, torch.Tensor)
                    else np.ascontiguousarray(arr.transpose(0, 2, 1))
                )
            if fold and k.endswith(FOLD_ONE):
                arr = (1.0 + arr).to(arr.dtype)
            out.update(self.emit(k, arr))

        for base, experts in per_expert.items():
            n = len(experts)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                values = [experts[i][proj] for i in range(n)]
                stacked = (
                    torch.stack(values, dim=0)
                    if isinstance(values[0], torch.Tensor)
                    else np.stack(values, axis=0)
                )
                out.update(self.emit(f"{base}.mlp.switch_mlp.{proj}.weight", stacked))
        return out

    def run(self, shard_bytes: int = 4_500_000_000):
        self.out.mkdir(parents=True, exist_ok=True)
        units = self.plan()
        order = ["top"] + sorted(u for u in units if u != "top")
        log.info("units: %d (top + %d layers)", len(order), len(order) - 1)

        buf: dict[str, np.ndarray] = {}
        buf_bytes = 0
        shards: list[str] = []

        def flush():
            nonlocal buf, buf_bytes
            if not buf:
                return
            name = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(
                {
                    k: (
                        torch.from_numpy(np.ascontiguousarray(v))
                        if isinstance(v, np.ndarray)
                        else v
                    )
                    for k, v in buf.items()
                },
                str(self.out / name),
                metadata={"format": "mlx"},
            )
            shards.append(name)
            log.info(
                "wrote %s (%.2f GiB, %d tensors)", name, buf_bytes / 2**30, len(buf)
            )
            buf = {}
            buf_bytes = 0

        for unit in order:
            packed = self.pack_unit(unit, units[unit])
            for k, v in packed.items():
                buf[k] = v
                buf_bytes += v.nbytes
            if buf_bytes >= shard_bytes:
                flush()

        flush()
        unused = [
            name for name in self.reader.tensors if name not in self.used_artifacts
        ]
        if unused:
            raise RuntimeError(
                f"{len(unused)} Stage A artifacts were not consumed; first={unused[:3]}"
            )
        self.write_metadata(shards)
        log.info(
            "done: injected=%d rtn=%d kept=%d",
            self.stats["injected"],
            self.stats["rtn"],
            self.stats["kept"],
        )

    def write_metadata(self, shards: list[str]):
        weights: dict[str, str] = {}
        total = 0
        for name in shards:
            from safetensors import safe_open as so

            with so(str(self.out / name), framework="pt") as f:
                for k in list(f.keys()):
                    weights[k] = name
            total += os.path.getsize(self.out / name)
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total},
                    "weight_map": dict(sorted(weights.items())),
                },
                indent=2,
            )
            + "\n"
        )
        cfg = json.loads((self.hf / "config.json").read_text())
        cfg["quantization"] = {
            "group_size": self.group_size,
            "bits": self.bits,
            "mode": self.mode,
        }
        (self.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        for extra in (
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        ):
            src = self.hf / extra
            if src.exists():
                shutil.copy2(src, self.out / extra)
        log.info(
            "metadata written: %d tensors across %d shards", len(weights), len(shards)
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--mode", default="mxfp4", choices=SUPPORTED_MODES)
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument(
        "--exclude",
        default=r"(^|\.)(gate|router)$|shared_expert_gate|e_score|(^|\.)ple($|\.)",
    )
    ap.add_argument("--storage-dtype", default="bfloat16")
    ap.add_argument("--shard-gib", type=float, default=4.2)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    default_group, default_bits = mode_defaults(args.mode)
    group_size = args.group_size or default_group
    bits = args.bits or default_bits
    packer = StreamingPacker(
        args.hf_path,
        args.calib,
        args.mlx_path,
        args.mode,
        bits,
        group_size,
        args.exclude,
        args.storage_dtype,
    )
    packer.run(int(args.shard_gib * 2**30))


if __name__ == "__main__":
    main()
