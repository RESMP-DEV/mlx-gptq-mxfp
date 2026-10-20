"""Calibration input capture for one decoder layer.

Two capture paths, both feeding CPU "stashes" of input rows:

1. nn.Linear modules (attention projections, dense MLPs, shared experts):
   forward pre-hooks, with same-input dedup (q/k/v and gate/up share stashes).

2. Fused MoE experts (transformers >= 5 stores experts as 3D parameters and
   loops ``F.linear(tokens, weight_3d[expert_idx])``): a TorchFunctionMode
   intercepts every F.linear call and matches the weight against registered
   per-expert slice data pointers. Each expert therefore accumulates exactly
   the tokens the router sent to it.

Stashes hold bf16 rows on CPU; Hessians (X^T X, fp32) are formed later on
whichever GPU solves that matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

log = logging.getLogger(__name__)

# 3D expert parameter names we understand, per the transformers fused layout.
FUSED_EXPERT_PARAMS = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")


class Stash:
    """Captured input rows for one logical weight matrix."""

    def __init__(self, name: str, in_features: int):
        self.name = name
        self.in_features = in_features
        self.chunks: list[torch.Tensor] = []
        self.tokens = 0

    def add(self, x: torch.Tensor):
        rows = x.detach().reshape(-1, x.shape[-1])
        if rows.shape[-1] != self.in_features:
            raise RuntimeError(
                f"stash {self.name}: got rows of dim {rows.shape[-1]}, "
                f"expected {self.in_features}"
            )
        self.chunks.append(rows.to("cpu", torch.bfloat16))
        self.tokens += rows.shape[0]

    def add_sampled(self, x: torch.Tensor, max_rows: int):
        """Add a deterministic, evenly spaced sample of input rows."""
        rows = x.detach().reshape(-1, x.shape[-1])
        if rows.shape[0] > max_rows:
            indices = torch.linspace(
                0,
                rows.shape[0] - 1,
                steps=max_rows,
                device=rows.device,
            ).long()
            rows = rows.index_select(0, indices)
        self.add(rows)

    def nbytes(self) -> int:
        return sum(c.numel() * 2 for c in self.chunks)

    @torch.no_grad()
    def hessian(self, device, chunk_rows: int = 65536) -> torch.Tensor:
        """H = X^T X in fp32 on `device`."""
        H = torch.zeros(self.in_features, self.in_features, dtype=torch.float32, device=device)
        for c in self.chunks:
            for i in range(0, c.shape[0], chunk_rows):
                x = c[i : i + chunk_rows].to(device, torch.float32, non_blocking=True)
                H.addmm_(x.T, x)
        return H

    def clear(self):
        self.chunks.clear()


@dataclass
class LinearTask:
    """A plain nn.Linear to quantize. `name` is layer-relative."""
    name: str
    module: torch.nn.Module
    stash_key: str  # owner stash (may belong to a sibling with the same input)


@dataclass
class FusedExpertTask:
    """One fused 3D expert parameter [E, out, in] to quantize per-expert."""
    module_name: str          # e.g. "mlp.experts"
    param_name: str           # e.g. "gate_up_proj"
    param: torch.nn.Parameter
    num_experts: int
    stash_keys: list[str] = field(default_factory=list)  # one per expert


class CaptureManager:
    """Owns all stashes and hooks for one decoder layer."""

    def __init__(self, layer: torch.nn.Module, exclude_regex, group_size: int):
        import re

        self.layer = layer
        self.stashes: dict[str, Stash] = {}
        self.linear_tasks: list[LinearTask] = []
        self.fused_tasks: list[FusedExpertTask] = []
        self._hooks = []
        self._ptr_registry: dict[int, str] = {}  # expert slice data_ptr -> stash key
        self._batch_seen: dict[int, tuple[torch.Tensor, str]] = {}
        self._fused_modules: dict[str, torch.nn.Module] = {}
        self._fused_input_stashes: dict[str, Stash] = {}
        self.fallback_experts: dict[str, str] = {}
        self._mode = None

        excl = re.compile(exclude_regex) if exclude_regex else None
        seen_fused_modules = set()

        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                if excl and excl.search(name):
                    log.debug("excluding %s (matches exclude regex)", name)
                    continue
                if mod.in_features % group_size != 0:
                    log.warning(
                        "skipping %s: in_features %d not divisible by group %d "
                        "(stays unquantized)", name, mod.in_features, group_size
                    )
                    continue
                self.linear_tasks.append(LinearTask(name, mod, stash_key=name))
                continue

            # Fused expert detection: module owning 3D params with known names.
            for pname, p in mod.named_parameters(recurse=False):
                if p.dim() == 3 and pname in FUSED_EXPERT_PARAMS:
                    if (name, pname) in seen_fused_modules:
                        continue
                    seen_fused_modules.add((name, pname))
                    E = p.shape[0]
                    if p.shape[-1] % group_size != 0:
                        log.warning(
                            "skipping fused %s.%s: in dim %d not divisible by group %d",
                            name, pname, p.shape[-1], group_size,
                        )
                        continue
                    task = FusedExpertTask(name, pname, p, E)
                    for e in range(E):
                        key = f"{name}.{pname}::e{e}"
                        task.stash_keys.append(key)
                        self.stashes[key] = Stash(key, p.shape[-1])
                        ptr = p.data_ptr() + e * p.stride(0) * p.element_size()
                        self._ptr_registry[ptr] = key
                    self.fused_tasks.append(task)
                    self._fused_modules[name] = mod
                    if pname in ("gate_up_proj", "gate_proj", "up_proj"):
                        self._fused_input_stashes.setdefault(
                            name, Stash(f"{name}::all_inputs", p.shape[-1])
                        )

        for t in self.linear_tasks:
            self.stashes.setdefault(t.name, Stash(t.name, t.module.in_features))

    # ---- nn.Linear pre-hooks with same-input dedup -------------------------

    def _make_hook(self, task: LinearTask):
        def hook(mod, args, kwargs=None):
            x = args[0] if args else kwargs["input"]
            seen = self._batch_seen.get(id(x))
            if seen is not None and seen[0] is x:
                # Same tensor already stashed by a sibling this batch (q/k/v,
                # gate/up, ...): share the sibling's stash instead of copying.
                task.stash_key = seen[1]
                return
            self._batch_seen[id(x)] = (x, task.stash_key)
            self.stashes[task.stash_key].add(x)
        return hook

    def _make_fused_input_hook(self, module_name: str):
        def hook(mod, args, kwargs=None):
            x = args[0] if args else kwargs["hidden_states"]
            # 64 rows from each of 64 standard calibration batches gives a
            # 4,096-row activation-aware fallback without materially changing
            # the memory profile of normal routed capture.
            self._fused_input_stashes[module_name].add_sampled(x, 64)

        return hook

    # ---- fused expert interception -----------------------------------------

    class _ExpertMode(TorchFunctionMode):
        def __init__(self, mgr):
            super().__init__()
            self.mgr = mgr

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is F.linear and len(args) >= 2:
                w = args[1]
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    key = self.mgr._ptr_registry.get(w.data_ptr())
                    if key is not None:
                        st = self.mgr.stashes[key]
                        if w.shape[-1] == st.in_features:
                            st.add(args[0])
            return func(*args, **kwargs)

    # ---- lifecycle -----------------------------------------------------------

    def __enter__(self):
        for t in self.linear_tasks:
            self._hooks.append(t.module.register_forward_pre_hook(self._make_hook(t)))
        for name, module in self._fused_modules.items():
            if name in self._fused_input_stashes:
                self._hooks.append(
                    module.register_forward_pre_hook(self._make_fused_input_hook(name))
                )
        if self._ptr_registry:
            self._mode = self._ExpertMode(self)
            self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self._mode is not None:
            self._mode.__exit__(*exc)
            self._mode = None
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._batch_seen.clear()
        return False

    def begin_batch(self):
        self._batch_seen.clear()

    # ---- diagnostics ----------------------------------------------------------

    def total_stash_bytes(self) -> int:
        return sum(s.nbytes() for s in self.stashes.values())

    def expert_token_stats(self):
        """Returns dict param -> (min, median, max, zero_count) of tokens/expert."""
        out = {}
        for t in self.fused_tasks:
            counts = sorted(self.stashes[k].tokens for k in t.stash_keys)
            zeros = sum(1 for c in counts if c == 0)
            out[f"{t.module_name}.{t.param_name}"] = (
                counts[0], counts[len(counts) // 2], counts[-1], zeros
            )
        return out

    @torch.no_grad()
    def backfill_unrouted_experts(self):
        """Calibrate never-routed experts from real inputs to their MoE block.

        Multimodal or otherwise specialized experts may receive no tokens in a
        text-only calibration corpus. For those experts only, use a bounded,
        deterministic sample of the block's real input activations. Down-proj
        inputs are then generated with that expert's original gate/up weights.
        This keeps the fallback activation-aware instead of using a zero
        Hessian and data-free rounding.
        """
        by_module: dict[str, dict[str, FusedExpertTask]] = {}
        for task in self.fused_tasks:
            by_module.setdefault(task.module_name, {})[task.param_name] = task

        backfilled = {}
        for module_name, tasks in by_module.items():
            source = self._fused_input_stashes.get(module_name)
            if source is None or source.tokens == 0:
                continue
            input_task = next(
                (
                    tasks[name]
                    for name in ("gate_up_proj", "gate_proj", "up_proj")
                    if name in tasks
                ),
                None,
            )
            if input_task is None:
                continue
            missing = [
                expert
                for expert, key in enumerate(input_task.stash_keys)
                if self.stashes[key].tokens == 0
            ]
            if not missing:
                continue

            for expert in missing:
                for task in tasks.values():
                    if task.param_name == "down_proj":
                        continue
                    stash = self.stashes[task.stash_keys[expert]]
                    for chunk in source.chunks:
                        stash.add(chunk)
                    self.fallback_experts[task.stash_keys[expert]] = (
                        "all_expert_inputs"
                    )

                down_task = tasks.get("down_proj")
                if down_task is not None:
                    down_stash = self.stashes[down_task.stash_keys[expert]]
                    if "gate_up_proj" in tasks:
                        gate_up = tasks["gate_up_proj"].param[expert]
                        for chunk in source.chunks:
                            gate, up = F.linear(
                                chunk.to(gate_up.device, gate_up.dtype), gate_up
                            ).chunk(2, dim=-1)
                            down_stash.add(F.silu(gate) * up)
                    elif "gate_proj" in tasks and "up_proj" in tasks:
                        gate_weight = tasks["gate_proj"].param[expert]
                        up_weight = tasks["up_proj"].param[expert]
                        for chunk in source.chunks:
                            x = chunk.to(gate_weight.device, gate_weight.dtype)
                            down_stash.add(
                                F.silu(F.linear(x, gate_weight))
                                * F.linear(x, up_weight)
                            )
                    if down_stash.tokens:
                        self.fallback_experts[down_task.stash_keys[expert]] = (
                            "all_expert_inputs"
                        )
                backfilled.setdefault(module_name, []).append(expert)
        return backfilled

    def check_capture_happened(self):
        """Abort early if fused experts never went through F.linear (e.g. a
        grouped-gemm experts implementation is active)."""
        for t in self.fused_tasks:
            if all(self.stashes[k].tokens == 0 for k in t.stash_keys):
                raise RuntimeError(
                    f"No tokens captured for fused experts {t.module_name}.{t.param_name}. "
                    "The model is probably not using the eager experts implementation; "
                    "load it with experts_implementation='eager'."
                )

    def drop_all(self):
        for s in self.stashes.values():
            s.clear()
        for s in self._fused_input_stashes.values():
            s.clear()
