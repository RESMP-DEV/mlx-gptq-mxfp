"""Stage B CLI (Apple Silicon): build a standard MLX model, then inject the
CUDA-calibrated q/scales/biases into its shards.

    python -m mlx_gptq.pack \
        --hf-path Qwen/Qwen3-30B-A3B --calib ./calib-qwen3-30b \
        --mlx-path ./Qwen3-30B-A3B-4bit-gptq --verify

Step 1 runs mlx_lm.convert (RTN) with the same bits/group so the output has
the exact tensor layout, config and tokenizer of a normal mlx-lm model.
Step 2 rewrites the shards, replacing every tensor we calibrated. Anything
without an artifact (embeddings, lm_head, tensors you excluded) stays RTN.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re

import numpy as np

from .artifacts import ArtifactReader
from .packing import pack_q

log = logging.getLogger("mlx_gptq")

ROUTER_SKIP = r"(^|\.)(gate|router)$|shared_expert_gate"


def make_predicate(
    skip_regex, head_bits, embed_bits, group_size, mode="affine", path_bits=None
):
    skip = re.compile(skip_regex) if skip_regex else None
    path_bits = path_bits or {}

    def predicate(path, module, *_):
        if skip and skip.search(path):
            return False
        if path in path_bits:
            return {"group_size": group_size, "bits": path_bits[path], "mode": mode}
        if path == "lm_head" or path.endswith(".lm_head"):
            return (
                {"group_size": group_size, "bits": head_bits, "mode": mode}
                if head_bits
                else False
            )
        if path.endswith("embed_tokens"):
            return (
                {"group_size": group_size, "bits": embed_bits, "mode": mode}
                if embed_bits
                else False
            )
        return True

    return predicate


def overrides_from_manifest(reader: ArtifactReader):
    """MLX module path -> bits, for artifacts that deviate from the global bits.

    Per-expert artifacts collapse onto one switch_mlp path, so their bits must
    agree across experts; stage A enforces that by construction.
    """
    global_bits = reader.manifest["bits"]
    out = {}
    for name, info in reader.tensors.items():
        if info["bits"] == global_bits:
            continue
        path = re.sub(r"\.experts\.\d+\.", ".switch_mlp.", name)
        path = path[: -len(".weight")] if path.endswith(".weight") else path
        prev = out.get(path)
        if prev is not None and prev != info["bits"]:
            raise RuntimeError(f"{path}: conflicting expert bits {prev} vs {info['bits']}")
        out[path] = info["bits"]
    return out


def _expert_artifact(mlx_key: str, e: int) -> str:
    # model.layers.L.mlp.switch_mlp.gate_proj.weight -> ...mlp.experts.{e}.gate_proj.weight
    return mlx_key.replace(".switch_mlp.", f".experts.{e}.")


def inject(mlx_path: str, reader: ArtifactReader):
    import mlx.core as mx

    index_path = os.path.join(mlx_path, "model.safetensors.index.json")
    shard_files = sorted(glob.glob(os.path.join(mlx_path, "model*.safetensors")))
    shard_files = [f for f in shard_files if not f.endswith(".index.json")]

    with open(os.path.join(mlx_path, "config.json")) as f:
        config = json.load(f)
    qcfg = config.get("quantization")
    if not qcfg:
        raise RuntimeError("converted model has no quantization config")

    def cfg_for(path):
        entry = qcfg.get(path)
        if isinstance(entry, dict):
            return (
                entry.get("group_size", qcfg["group_size"]),
                entry.get("bits", qcfg["bits"]),
                entry.get("mode", qcfg.get("mode", "affine")),
            )
        return qcfg["group_size"], qcfg["bits"], qcfg.get("mode", "affine")

    used = set()
    injected, rtn_kept = [], []

    for shard in shard_files:
        weights = mx.load(shard)
        changed = False
        for key in list(weights.keys()):
            if not key.endswith(".weight") or f"{key[:-7]}.scales" not in weights:
                continue
            base = key[:-7]  # strip ".weight"
            wq_old = weights[key]
            s_old = weights[f"{base}.scales"]
            group_size, bits, mode = cfg_for(base)

            if wq_old.ndim == 3 and ".switch_mlp." in key:
                E = wq_old.shape[0]
                arts = [_expert_artifact(key, e) for e in range(E)]
                if not all(reader.has(a) for a in arts):
                    missing = [a for a in arts if not reader.has(a)]
                    if len(missing) < E:
                        raise RuntimeError(
                            f"{key}: only some experts have artifacts (e.g. {missing[0]})"
                        )
                    rtn_kept.append(key)
                    continue
                qs, ss, bs = [], [], []
                for a in arts:
                    q, s, b, info = reader.get(a)
                    artifact_mode = info.get("mode", "affine")
                    if (
                        info["bits"] != bits
                        or info["group_size"] != group_size
                        or artifact_mode != mode
                    ):
                        raise RuntimeError(
                            f"{a}: artifact is {artifact_mode}/{info['bits']}b/"
                            f"g{info['group_size']} but model config expects "
                            f"{mode}/{bits}b/g{group_size}; re-run convert "
                            "with matching settings or fix overrides"
                        )
                    qs.append(q.numpy())
                    ss.append(s.numpy())
                    if b is not None:
                        bs.append(b.float().numpy())
                    used.add(a)
                q_all = np.stack(qs)
                s_all = np.stack(ss)
                b_all = np.stack(bs) if bs else None
            elif wq_old.ndim == 2 and reader.has(key):
                q, s, b, info = reader.get(key)
                artifact_mode = info.get("mode", "affine")
                if (
                    info["bits"] != bits
                    or info["group_size"] != group_size
                    or artifact_mode != mode
                ):
                    raise RuntimeError(
                        f"{key}: artifact {artifact_mode}/{info['bits']}b/"
                        f"g{info['group_size']} vs config {mode}/{bits}b/"
                        f"g{group_size}"
                    )
                q_all = q.numpy()
                s_all = s.numpy()
                b_all = None if b is None else b.float().numpy()
                used.add(key)
            else:
                rtn_kept.append(key)
                continue

            words = pack_q(q_all, bits)
            if words.shape != wq_old.shape:
                raise RuntimeError(
                    f"{key}: packed shape {words.shape} != converted shape {wq_old.shape}"
                )
            if s_all.shape != tuple(s_old.shape):
                raise RuntimeError(
                    f"{base}.scales: artifact shape {s_all.shape} != {tuple(s_old.shape)}"
                )
            weights[key] = mx.array(words)
            weights[f"{base}.scales"] = mx.array(s_all).astype(s_old.dtype)
            if mode == "affine":
                if b_all is None or f"{base}.biases" not in weights:
                    raise RuntimeError(f"{base}: affine artifact is missing biases")
                weights[f"{base}.biases"] = mx.array(b_all).astype(s_old.dtype)
            elif b_all is not None:
                raise RuntimeError(f"{base}: {mode} artifact unexpectedly has biases")
            injected.append(key)
            changed = True

        if changed:
            # mx.load arrays are mmap-backed: never save over the file they
            # still read from (corrupts every tensor not explicitly replaced).
            tmp = shard.replace(".safetensors", ".inject-tmp.safetensors")
            mx.save_safetensors(tmp, weights, metadata={"format": "mlx"})
            os.replace(tmp, shard)
        del weights
        log.info("shard %s done", os.path.basename(shard))

    unused = [a for a in reader.tensors if a not in used]
    log.info("injected %d tensors; %d stayed RTN", len(injected), len(rtn_kept))
    if rtn_kept:
        preview = [k for k in rtn_kept if "layers.0." in k or "layers." not in k]
        log.info("RTN-kept (unique/preview): %s", preview[:20])
    if unused:
        raise RuntimeError(
            f"{len(unused)} artifacts were never consumed (naming mismatch?), "
            f"e.g. {unused[:5]}"
        )
    if not injected:
        raise RuntimeError("nothing was injected — check artifact naming vs model")
    # index file stays valid: shapes and dtypes are unchanged.
    _ = index_path
    return injected, rtn_kept


def verify(mlx_path: str, prompt: str = "The capital of France is"):
    from mlx_lm import generate, load

    model, tokenizer = load(mlx_path)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=40)
    log.info("verify generate: %s -> %s", prompt, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-path", required=True, help="original HF model (id or dir)")
    ap.add_argument("--calib", required=True, help="Stage A artifact dir")
    ap.add_argument("--mlx-path", required=True, help="output MLX model dir")
    ap.add_argument("--skip-convert", action="store_true",
                    help="mlx-path already holds a converted RTN model; only inject")
    ap.add_argument("--router-skip", default=ROUTER_SKIP,
                    help="regex of module paths to leave unquantized during convert")
    ap.add_argument("--head-bits", type=int, default=None,
                    help="RTN bits for lm_head (default: global bits)")
    ap.add_argument("--embed-bits", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--verify", action="store_true", help="load and generate afterwards")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    reader = ArtifactReader(args.calib)
    m = reader.manifest
    mode = m.get("mode", "affine")
    log.info("artifacts: %s | %s %db g%d | %d tensors",
             m["model"], mode, m["bits"], m["group_size"], len(reader.tensors))

    if not args.skip_convert:
        from mlx_lm.convert import convert

        convert(
            args.hf_path,
            mlx_path=args.mlx_path,
            quantize=True,
            q_group_size=m["group_size"],
            q_bits=m["bits"],
            q_mode=mode,
            dtype=m["storage_dtype"],
            quant_predicate=make_predicate(
                args.router_skip,
                args.head_bits,
                args.embed_bits,
                m["group_size"],
                mode=mode,
                path_bits=overrides_from_manifest(reader),
            ),
            trust_remote_code=args.trust_remote_code,
        )
        log.info("RTN convert complete: %s", args.mlx_path)

    inject(args.mlx_path, reader)
    log.info("calibrated model ready: %s", args.mlx_path)

    if args.verify:
        verify(args.mlx_path)


if __name__ == "__main__":
    main()
