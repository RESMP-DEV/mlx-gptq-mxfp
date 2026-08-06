"""Layer-sequential GPTQ pipeline (Stage A).

The model is never fully materialized: a meta-device skeleton is built from
the config, and each decoder layer's weights are streamed from the raw
safetensors shards right before it is processed, then freed. This keeps RAM
bounded to (activations + one layer + stashes) regardless of model size.

Per layer:
  1. materialize weights on the forward device
  2. forward all calibration batches with capture hooks -> CPU stashes
  3. solve GPTQ per weight matrix (experts batched, spread across GPUs)
  4. write dequantized weights back into the layer
  5. re-forward to produce the *quantized* activations for the next layer
  6. append artifacts, free everything
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
from tqdm import tqdm

from .capture import CaptureManager, FusedExpertTask, LinearTask
from .gptq import solve_gptq
from .grid import dequantize_artifact

log = logging.getLogger(__name__)

DEFAULT_EXCLUDE = r"(^|\.)gate$|(^|\.)router($|\.)|shared_expert_gate|e_score"


# --------------------------------------------------------------------------
# Raw checkpoint access
# --------------------------------------------------------------------------

class RawShardIndex:
    """Name -> shard lookup over a HF model directory's safetensors files."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"no safetensors found in {model_dir}")
            from safetensors import safe_open

            with safe_open(single, framework="pt") as f:
                names = list(f.keys())
            self.weight_map = {n: "model.safetensors" for n in names}
        self._open = {}

    def _file(self, fname):
        if fname not in self._open:
            from safetensors import safe_open

            self._open[fname] = safe_open(
                os.path.join(self.dir, fname), framework="pt"
            )
        return self._open[fname]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._file(self.weight_map[name]).get_tensor(name)

    def names_with_prefix(self, prefix: str):
        return [n for n in self.weight_map if n.startswith(prefix)]


def materialize_module(module: torch.nn.Module, prefix: str, shards: RawShardIndex,
                       device, dtype) -> None:
    """Fill a meta-device module's params/buffers from raw checkpoint tensors.

    Handles the per-expert -> fused 3D conversion used by transformers >= 5:
        runtime  X.gate_up_proj [E, 2I, H] <- raw X.{e}.gate_proj / X.{e}.up_proj
        runtime  X.down_proj    [E, H, I]  <- raw X.{e}.down_proj
    """
    sd = {}
    for pname, p in module.named_parameters():
        raw = f"{prefix}.{pname}"
        if shards.has(raw):
            t = shards.get(raw)
            if t.is_floating_point():
                t = t.to(dtype)
            sd[pname] = t.to(device)
            continue
        m = re.match(r"^(.*)\.(gate_up_proj|gate_proj|up_proj|down_proj)$", pname)
        if m and p.dim() == 3:
            base, kind = m.groups()
            E = p.shape[0]
            fused = torch.empty(p.shape, dtype=dtype, device=device)
            for e in range(E):
                if kind == "gate_up_proj":
                    g = shards.get(f"{prefix}.{base}.{e}.gate_proj.weight").to(dtype)
                    u = shards.get(f"{prefix}.{base}.{e}.up_proj.weight").to(dtype)
                    I = g.shape[0]
                    fused[e, :I] = g
                    fused[e, I:] = u
                else:
                    fused[e] = shards.get(f"{prefix}.{base}.{e}.{kind}.weight").to(dtype)
            sd[pname] = fused
            continue
        raise KeyError(f"cannot materialize parameter {prefix}.{pname} from checkpoint")

    for bname, b in module.named_buffers():
        if b.is_meta:
            raw = f"{prefix}.{bname}"
            if shards.has(raw):
                sd[bname] = shards.get(raw).to(device)
            else:
                raise KeyError(f"meta buffer {prefix}.{bname} not in checkpoint")

    module.load_state_dict(sd, strict=False, assign=True)
    # sanity: nothing left on meta
    for n, p in module.named_parameters():
        if p.is_meta:
            raise RuntimeError(f"parameter {prefix}.{n} still on meta after materialize")


def free_module(module: torch.nn.Module):
    """Release a layer's weights by putting it back on meta."""
    module.to_empty(device="meta")


# --------------------------------------------------------------------------
# Layer-0 input catcher
# --------------------------------------------------------------------------

class _CatchDone(Exception):
    pass


class Catcher(torch.nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def forward(self, hidden_states, **kwargs):
        self.store.append((hidden_states.detach().cpu(), kwargs))
        raise _CatchDone


def _move_kwargs(kwargs, device):
    def mv(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, tuple):
            return tuple(mv(x) for x in v)
        return v

    return {k: mv(v) for k, v in kwargs.items()}


# --------------------------------------------------------------------------
# Solve planning
# --------------------------------------------------------------------------

def artifact_names_for_fused(layer_prefix: str, task: FusedExpertTask, expert: int):
    """Raw-checkpoint style artifact names for one expert of a fused param.

    gate_up_proj rows are [gate; up] (verified: forward chunks output in that
    order), so it splits into two artifacts.
    """
    base = f"{layer_prefix}.{task.module_name}"
    if task.param_name == "gate_up_proj":
        return [
            f"{base}.{expert}.gate_proj.weight",
            f"{base}.{expert}.up_proj.weight",
        ]
    return [f"{base}.{expert}.{task.param_name}.weight"]


def _expert_chunk_size(C: int, R: int, vram_gb: float) -> int:
    """How many experts fit in one batched GPTQ solve within a VRAM budget."""
    per = (3 * C * C + 3 * R * C) * 4  # H, U, chol temps + W copies (fp32)
    budget = vram_gb * (1 << 30)
    return max(1, int(budget // per))


class SolveResult:
    def __init__(self):
        self.tensors = {}   # artifact name -> (q, scales, biases) on CPU
        self.meta = {}      # artifact name -> info
        self.lock = threading.Lock()

    def add(self, name, q, s, b, info):
        with self.lock:
            self.tensors[name] = (q.cpu(), s.cpu(), None if b is None else b.cpu())
            self.meta[name] = info


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

class Pipeline:
    def __init__(self, model, shards: RawShardIndex, args):
        self.model = model
        self.shards = shards
        self.args = args
        self.dev0 = args.devices[0]
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.storage_dtype = self.dtype
        inner = model
        for attr in args.layers_attr.split(".")[:-1]:
            inner = getattr(inner, attr)
        self.inner = inner
        self.layers = getattr(inner, args.layers_attr.split(".")[-1])
        self.layer_prefix_fmt = args.layers_attr + ".{}"

    # ---- initial activations -------------------------------------------------

    @torch.no_grad()
    def catch_layer0_inputs(self, samples: torch.Tensor, batch_size: int):
        """Returns list of (hidden_cpu, kwargs) per forward batch."""
        # Only the input embeddings are needed before layer 0 (rotary inv_freq
        # buffers are real already; the final norm / lm_head are never reached).
        emb = self.model.get_input_embeddings()
        emb_name = None
        for n, m in self.model.named_modules():
            if m is emb:
                emb_name = n
                break
        materialize_module(emb, emb_name, self.shards, self.dev0, self.dtype)

        store = []
        orig = self.layers[0]
        self.layers[0] = Catcher(store)
        try:
            for i in range(0, samples.shape[0], batch_size):
                ids = samples[i : i + batch_size].to(self.dev0)
                try:
                    self.model(input_ids=ids, use_cache=False)
                except _CatchDone:
                    pass
        finally:
            self.layers[0] = orig
        free_module(emb)
        if len(store) == 0:
            raise RuntimeError("catcher captured nothing; layers_attr wrong?")
        return store

    # ---- per-layer processing --------------------------------------------------

    @torch.no_grad()
    def forward_layer(self, layer, batches, capture_mgr=None):
        """Run all batches through `layer`; returns new hidden states (CPU)."""
        outs = []
        for hid_cpu, kwargs in batches:
            if capture_mgr is not None:
                capture_mgr.begin_batch()
            hs = hid_cpu.to(self.dev0, self.dtype)
            kw = _move_kwargs(kwargs, self.dev0)
            out = layer(hs, **kw)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out.detach().cpu())
        return outs

    def _bits_for(self, artifact_name: str):
        for pattern, bits in self.args.overrides:
            if pattern.search(artifact_name):
                return bits
        return self.args.bits

    @torch.no_grad()
    def solve_layer(self, layer_idx: int, mgr: CaptureManager) -> SolveResult:
        prefix = self.layer_prefix_fmt.format(layer_idx)
        result = SolveResult()
        devices = list(self.args.devices)
        dev_q: Queue = Queue()
        for d in devices:
            dev_q.put(d)

        hess_cache: dict[str, tuple[str, torch.Tensor]] = {}
        hess_lock = threading.Lock()

        def get_hessian(stash_key: str, device):
            with hess_lock:
                hit = hess_cache.get(stash_key)
                if hit is not None and hit[0] == str(device):
                    return hit[1]
            H = mgr.stashes[stash_key].hessian(device)
            with hess_lock:
                hess_cache[stash_key] = (str(device), H)
            return H

        def run_linear(task: LinearTask):
            device = dev_q.get()
            try:
                name = f"{prefix}.{task.name}.weight"
                bits = self._bits_for(name)
                W = task.module.weight.detach().to(device, torch.float32).unsqueeze(0)
                H = get_hessian(task.stash_key, device).unsqueeze(0)
                q, s, b, W_dq, rel = solve_gptq(
                    W, H, bits, self.args.group_size,
                    damp=self.args.damp, clip=self.args.clip,
                    storage_dtype=self.storage_dtype,
                    mode=self.args.mode,
                    mxfp_algorithm=self.args.mxfp_algorithm,
                )
                task.module.weight.data.copy_(W_dq[0].to(task.module.weight.dtype))
                result.add(name, q[0], s[0], None if b is None else b[0], {
                    "bits": bits, "group_size": self.args.group_size,
                    "mode": self.args.mode,
                    "kind": "linear",
                    "out_features": W.shape[1], "in_features": W.shape[2],
                    "tokens": mgr.stashes[task.stash_key].tokens,
                    "rel_err": round(float(rel[0]), 6),
                })
            finally:
                dev_q.put(device)

        def run_fused(task: FusedExpertTask):
            E = task.num_experts
            p = task.param
            R, C = p.shape[1], p.shape[2]
            # gate/up halves of a fused param always share bits (they land in
            # one stacked MLX tensor); the first expert's first artifact decides.
            bits = self._bits_for(artifact_names_for_fused(prefix, task, 0)[0])
            chunk = _expert_chunk_size(C, R, self.args.vram_gb)
            spans = [(e0, min(e0 + chunk, E)) for e0 in range(0, E, chunk)]

            def one_span(span):
                e0, e1 = span
                device = dev_q.get()
                try:
                    Hs, Ws = [], []
                    for e in range(e0, e1):
                        Hs.append(mgr.stashes[task.stash_keys[e]].hessian(device))
                        Ws.append(p[e].detach().to(device, torch.float32))
                    H = torch.stack(Hs); del Hs
                    W = torch.stack(Ws); del Ws
                    q, s, b, W_dq, rel = solve_gptq(
                        W, H, bits, self.args.group_size,
                        damp=self.args.damp, clip=self.args.clip,
                        storage_dtype=self.storage_dtype,
                        mode=self.args.mode,
                        mxfp_algorithm=self.args.mxfp_algorithm,
                    )
                    del H
                    p.data[e0:e1].copy_(W_dq.to(p.dtype).to(p.device))
                    for i, e in enumerate(range(e0, e1)):
                        names = artifact_names_for_fused(prefix, task, e)
                        if len(names) == 1:
                            halves = [(q[i], s[i], None if b is None else b[i])]
                        else:
                            halves = [
                                (
                                    q[i, : R // 2],
                                    s[i, : R // 2],
                                    None if b is None else b[i, : R // 2],
                                ),
                                (
                                    q[i, R // 2 :],
                                    s[i, R // 2 :],
                                    None if b is None else b[i, R // 2 :],
                                ),
                            ]
                        for nm, (qq, ss, bb) in zip(names, halves):
                            result.add(nm, qq, ss, bb, {
                                "bits": bits, "group_size": self.args.group_size,
                                "mode": self.args.mode,
                                "kind": "expert",
                                "out_features": qq.shape[0], "in_features": C,
                                "tokens": mgr.stashes[task.stash_keys[e]].tokens,
                                "rel_err": round(float(rel[i]), 6),
                            })
                    del W, q, s
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                finally:
                    dev_q.put(device)

            with ThreadPoolExecutor(max_workers=len(devices)) as ex:
                list(ex.map(one_span, spans))

        # Solve fused experts (the bulk) first, then the dense linears.
        for task in mgr.fused_tasks:
            run_fused(task)
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            list(ex.map(run_linear, mgr.linear_tasks))
        return result

    @torch.no_grad()
    def apply_precomputed(self, layer, layer_idx: int, reader) -> None:
        """Resume path: write dequantized artifact weights into a layer."""
        prefix = self.layer_prefix_fmt.format(layer_idx)
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                art = f"{prefix}.{name}.weight"
                if reader.has(art):
                    q, s, b, info = reader.get(art)
                    G = info["group_size"]
                    dq = dequantize_artifact(
                        q, s, b, info.get("mode", "affine"), G
                    )
                    mod.weight.data.copy_(dq.reshape(q.shape).to(mod.weight.dtype).to(mod.weight.device))
            for pname, p in mod.named_parameters(recurse=False):
                if p.dim() == 3 and pname in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
                    E, R, C = p.shape
                    for e in range(E):
                        if pname == "gate_up_proj":
                            arts = [f"{prefix}.{name}.{e}.gate_proj.weight",
                                    f"{prefix}.{name}.{e}.up_proj.weight"]
                            rows = [(0, R // 2), (R // 2, R)]
                        else:
                            arts = [f"{prefix}.{name}.{e}.{pname}.weight"]
                            rows = [(0, R)]
                        for art, (r0, r1) in zip(arts, rows):
                            if not reader.has(art):
                                continue
                            q, s, b, info = reader.get(art)
                            G = info["group_size"]
                            dq = dequantize_artifact(
                                q, s, b, info.get("mode", "affine"), G
                            )
                            p.data[e, r0:r1].copy_(dq.reshape(q.shape).to(p.dtype).to(p.device))

    # ---- orchestration -------------------------------------------------------

    def run(self, samples: torch.Tensor, writer, reader_for_resume=None):
        t_start = time.time()
        batches = self.catch_layer0_inputs(samples, self.args.batch_size)
        log.info("caught %d forward batches for layer 0", len(batches))

        for L in tqdm(range(len(self.layers)), desc="layers"):
            layer = self.layers[L]
            prefix = self.layer_prefix_fmt.format(L)
            t0 = time.time()
            materialize_module(layer, prefix, self.shards, self.dev0, self.dtype)

            if writer.layer_done(L):
                self.apply_precomputed(layer, L, reader_for_resume)
                new_batches = self.forward_layer(layer, batches)
                for i, (___, kw) in enumerate(batches):
                    batches[i] = (new_batches[i], kw)
                free_module(layer)
                log.info("layer %d: resumed from artifacts (%.1fs)", L, time.time() - t0)
                continue

            mgr = CaptureManager(layer, self.args.exclude, self.args.group_size)
            n_lin = len(mgr.linear_tasks)
            n_exp = sum(t.num_experts for t in mgr.fused_tasks)
            with mgr:
                self.forward_layer(layer, batches, capture_mgr=mgr)
            if mgr.fused_tasks:
                mgr.check_capture_happened()
                for pname, (mn, med, mx, zeros) in mgr.expert_token_stats().items():
                    log.info("layer %d %s tokens/expert min=%d med=%d max=%d zero=%d",
                             L, pname, mn, med, mx, zeros)
                    if zeros:
                        log.warning("layer %d %s: %d experts saw ZERO tokens "
                                    "(they fall back to data-free rounding); "
                                    "consider more calibration samples", L, pname, zeros)
            t_cap = time.time()

            result = self.solve_layer(L, mgr)
            mgr.drop_all()
            t_solve = time.time()

            new_batches = self.forward_layer(layer, batches)
            for i, (___, kw) in enumerate(batches):
                batches[i] = (new_batches[i], kw)
            free_module(layer)
            if str(self.dev0).startswith("cuda"):
                torch.cuda.empty_cache()

            writer.write_layer(L, result.tensors, result.meta)
            errs = [m["rel_err"] for m in result.meta.values()]
            log.info(
                "layer %d done: %d linears + %d experts | rel_err med=%.4f max=%.4f "
                "| capture %.1fs solve %.1fs total %.1fs",
                L, n_lin, n_exp,
                sorted(errs)[len(errs) // 2] if errs else -1,
                max(errs) if errs else -1,
                t_cap - t0, t_solve - t_cap, time.time() - t0,
            )

        log.info("all %d layers done in %.1f min", len(self.layers), (time.time() - t_start) / 60)
