"""Faithful ContextOnly-cache replay of the DFlash2 draft forward pass.

The MLX runtime (dflash_mlx) advances a per-layer ContextOnlyDraftKVCache
that stores only target-context keys, trims to a sink plus window, and uses
absolute logical positions for RoPE and the sliding-window mask. The training
replay must reproduce that contract exactly; otherwise the objective is
evaluated on a different function than the one that will serve traffic.

A fidelity gate (replay_fidelity) compares reconstructed hidden states against
the runtime-captured selector hidden states so a mismatch fails the run instead
of silently training on the wrong forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

CONTEXT_SINK = 64
CONTEXT_WINDOW = 1024


def _rope(model, x, positions):
    # x is [batch, heads, length, head_dim]; compute cos/sin from logical length.
    batch, _, length, _ = x.shape
    dummy = x.new_empty(batch, length, 1)
    cos, sin = model.rotary_emb(dummy, positions.reshape(1, -1))
    return apply_rotary_pos_emb(x, x, cos, sin)[0]


def new_cache(num_layers: int) -> list[dict]:
    return [
        {"keys": None, "values": None, "positions": None, "offset": 0}
        for _ in range(num_layers)
    ]


def _trim(cache):
    keys = cache["keys"]
    if keys is None:
        return
    if keys.shape[2] <= CONTEXT_SINK + CONTEXT_WINDOW:
        return
    cache["keys"] = torch.cat(
        [keys[:, :, :CONTEXT_SINK], keys[:, :, -CONTEXT_WINDOW:]], dim=2
    )
    cache["values"] = torch.cat(
        [cache["values"][:, :, :CONTEXT_SINK], cache["values"][:, :, -CONTEXT_WINDOW:]],
        dim=2,
    )
    cache["positions"] = torch.cat(
        [cache["positions"][:CONTEXT_SINK], cache["positions"][-CONTEXT_WINDOW:]]
    )


def _spans(ctx_len: int, cache_len: int) -> list[tuple[int, int]]:
    max_len = CONTEXT_SINK + CONTEXT_WINDOW
    if cache_len == 0:
        if ctx_len <= max_len:
            return [(0, ctx_len)]
        spans = []
        sink_end = min(CONTEXT_SINK, ctx_len)
        if sink_end > 0:
            spans.append((0, sink_end))
        tail_start = max(sink_end, ctx_len - CONTEXT_WINDOW)
        if tail_start < ctx_len:
            spans.append((tail_start, ctx_len))
        return spans
    if ctx_len <= CONTEXT_WINDOW:
        return [(0, ctx_len)]
    return [(ctx_len - CONTEXT_WINDOW, ctx_len)]


def _layer_attention(layer, hidden, context, cache, model):
    batch, length, _ = hidden.shape
    heads = layer.config.num_attention_heads
    kv_heads = layer.config.num_key_value_heads
    head_dim = layer.head_dim
    ctx_len = context.shape[1]
    cache_len = 0 if cache["keys"] is None else cache["keys"].shape[2]
    spans = _spans(ctx_len, cache_len)
    if spans:
        selected = torch.cat([context[:, s:e] for s, e in spans], dim=1)
    else:
        selected = context[:, :0]

    q = layer.q_norm(layer.q_proj(hidden).view(batch, length, heads, head_dim))
    q = q.transpose(1, 2)
    ctx_keys = layer.k_norm(
        layer.k_proj(selected).view(batch, selected.shape[1], kv_heads, head_dim)
    ).transpose(1, 2)
    ctx_values = layer.v_proj(selected).view(
        batch, selected.shape[1], kv_heads, head_dim
    ).transpose(1, 2)
    noise_keys = layer.k_norm(
        layer.k_proj(hidden).view(batch, length, kv_heads, head_dim)
    ).transpose(1, 2)
    noise_values = layer.v_proj(hidden).view(
        batch, length, kv_heads, head_dim
    ).transpose(1, 2)

    offset = cache["offset"]
    if spans:
        span_positions = torch.cat(
            [
                torch.arange(offset + s, offset + e, dtype=torch.long, device=hidden.device)
                for s, e in spans
            ]
        )
    else:
        span_positions = torch.empty(0, dtype=torch.long, device=hidden.device)
    query_positions = torch.arange(
        offset + ctx_len, offset + ctx_len + length, dtype=torch.long, device=hidden.device
    )

    q = _rope(model, q, query_positions)
    ctx_keys = _rope(model, ctx_keys, span_positions)
    noise_keys = _rope(model, noise_keys, query_positions)

    previous_keys = cache["keys"]
    previous_values = cache["values"]
    if ctx_keys.shape[2] > 0:
        cache["keys"] = (
            ctx_keys.detach()
            if previous_keys is None
            else torch.cat([previous_keys, ctx_keys.detach()], dim=2)
        )
        cache["values"] = (
            ctx_values.detach()
            if previous_values is None
            else torch.cat([previous_values, ctx_values.detach()], dim=2)
        )
        cache["positions"] = (
            span_positions
            if cache["positions"] is None
            else torch.cat([cache["positions"], span_positions])
        )
    else:
        cache["keys"] = torch.empty_like(ctx_keys)
        cache["values"] = torch.empty_like(ctx_values)
        cache["positions"] = torch.empty(0, dtype=torch.long, device=hidden.device)
    cache["offset"] = offset + ctx_len
    _trim(cache)

    retained_previous = max(0, cache["keys"].shape[2] - ctx_keys.shape[2])
    context_keys = (
        ctx_keys
        if previous_keys is None
        else torch.cat(
            [previous_keys[:, :, -retained_previous:, :], ctx_keys], dim=2
        )
    )
    context_values = (
        ctx_values
        if previous_values is None
        else torch.cat(
            [previous_values[:, :, -retained_previous:, :], ctx_values], dim=2
        )
    )
    keys = torch.cat([context_keys, noise_keys], dim=2)
    values = torch.cat([context_values, noise_values], dim=2)
    key_positions = torch.cat([cache["positions"], query_positions])
    if keys.shape[2] != key_positions.numel():
        raise RuntimeError(
            f"replay key count {keys.shape[2]} does not match "
            f"position count {key_positions.numel()}"
        )

    mask = (query_positions[:, None] >= key_positions[None, :]) & (
        query_positions[:, None] < key_positions[None, :] + layer.sliding_window
    )
    repeat = heads // kv_heads
    if repeat > 1:
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(
        q, keys, values, attn_mask=mask[None, None], scale=layer.scaling
    )
    return layer.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


def _layer_forward(layer, hidden, context, cache, model):
    residual = hidden
    x = layer.input_layernorm(hidden)
    kernel = None
    if layer.attention_conv is not None:
        x, kernel = layer.attention_conv.prepare(x)
    x = _layer_attention(layer.self_attn, x, context, cache, model)
    if layer.attention_conv is not None:
        x = layer.attention_conv.finish(x, kernel)
    x = residual + x
    residual = x
    x = layer.post_attention_layernorm(x)
    kernel = None
    if layer.mlp_conv is not None:
        x, kernel = layer.mlp_conv.prepare(x)
    x = layer.mlp(x)
    if layer.mlp_conv is not None:
        x = layer.mlp_conv.finish(x, kernel)
    return residual + x


def replay_context(model, context, noise, caches):
    """Run one draft block against a ContextOnly cache and return normed hidden."""
    hidden = noise
    for layer, cache in zip(model.layers, caches, strict=True):
        hidden = _layer_forward(layer, hidden, context, cache, model)
    return model.norm(hidden)[:, 1:, :]


def draft_context_for(model, payload):
    """Return the runtime draft_context (already hidden_norm(fc(raw_context)))."""
    context = payload["context"]
    raw = payload.get("raw_context")
    if (
        raw is not None
        and raw.shape[1] == context.shape[1]
        and raw.shape[-1] == model.fc.in_features
    ):
        return model.hidden_norm(model.fc(raw))
    return context


class PromptCaches:
    """Per-prompt ContextOnly caches advanced only in capture order.

    The runtime cache is sequential: each committed cycle extends the cache by
    the context length that the next cycle supplies. Sampling that visits a
    prompt's cycles out of capture order cannot reuse a mutated cache, so the
    per-prompt cache is rebuilt whenever ordering breaks.
    """

    def __init__(self, num_layers: int, prompt_order: dict[int, list[int]]):
        self.num_layers = num_layers
        self.position = {
            int(pid): {cycle: i for i, cycle in enumerate(cycles)}
            for pid, cycles in prompt_order.items()
        }
        self.state: dict[int, tuple[list[dict], int]] = {}

    def replay(self, model, payload, prompt_id: int, cycle_index: int):
        order = self.position[int(prompt_id)][int(cycle_index)]
        caches, last = self.state.get(int(prompt_id), (None, -1))
        if caches is None or last + 1 != order:
            caches = new_cache(self.num_layers)
        context = draft_context_for(model, payload)
        hidden = replay_context(model, context, payload["noise_embedding"], caches)
        self.state[int(prompt_id)] = (caches, order)
        return hidden


def _prompt_order(dataset):
    order = {}
    for index, prompt_id in enumerate(dataset.prompt_ids):
        order.setdefault(int(prompt_id), []).append(index)
    return order


def _fidelity_plan(dataset, limit: int) -> list[int]:
    """Choose contiguous prefixes across several prompts and cache ages."""
    groups = list(_prompt_order(dataset).values())
    if not groups:
        return []
    group_count = min(len(groups), max(1, limit // 4))
    depth = -(-limit // group_count)
    plan: list[int] = []
    for group in groups[:group_count]:
        plan.extend(group[:depth])
    return sorted(plan[:limit])


def replay_fidelity(model, dataset, device, *, limit=32, threshold=0.97):
    """Median cosine between cached replay and runtime-captured hidden states.

    Raises RuntimeError when the median falls below threshold, so a replay that
    diverges from serving fails the run instead of training on the wrong
    forward pass.
    """
    caches = PromptCaches(len(model.layers), _prompt_order(dataset))
    cosines = []
    with torch.no_grad():
        for index in _fidelity_plan(dataset, min(limit, len(dataset))):
            payload, prompt_id = dataset[index]
            local = {
                "context": payload["context"].to(device),
                "raw_context": (
                    payload["raw_context"].to(device)
                    if payload.get("raw_context") is not None
                    else None
                ),
                "noise_embedding": payload["noise_embedding"].to(device),
            }
            captured = payload["selector_hidden"].to(device).float()
            got = caches.replay(
                model, local, int(prompt_id), index
            ).float()
            if got.shape != captured.shape:
                continue
            cosines.append(
                float(
                    F.cosine_similarity(
                        got.flatten(), captured.flatten(), dim=0
                    )
                )
            )
    if not cosines:
        raise RuntimeError("replay fidelity produced no comparable cycles")
    cosines.sort()
    median = cosines[len(cosines) // 2]
    if median < threshold:
        raise RuntimeError(
            f"draft replay fidelity {median:.4f} below threshold {threshold:.4f}; "
            "the training forward pass does not match the runtime"
        )
    return median, len(cosines)
