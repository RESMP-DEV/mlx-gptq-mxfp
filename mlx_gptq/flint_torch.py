"""Eager PyTorch Flint skeleton for layer-sequential GPTQ calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class FlintConfig:
    model_type: str = "flint"
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 150144
    num_experts: int = 256
    num_experts_per_tok: int = 6
    sliding_window: int = 513
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    layer_types: list[str] | None = None
    no_rope_layer_indices: list[int] | None = None
    mlp_only_layers: list[int] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in params.items() if k in fields})


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    )
    angles = positions[:, None].to(x.device, torch.float32) * inv_freq[None, :]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


class Attention(torch.nn.Module):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.skip_rope = layer_idx in set(config.no_rope_layer_indices or [])
        self.rope_theta = config.rope_theta

        hidden = config.hidden_size
        self.q_proj = torch.nn.Linear(
            hidden, self.num_heads * self.head_dim, config.attention_bias
        )
        self.k_proj = torch.nn.Linear(
            hidden, self.num_kv_heads * self.head_dim, config.attention_bias
        )
        self.v_proj = torch.nn.Linear(
            hidden, self.num_kv_heads * self.head_dim, config.attention_bias
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, hidden, config.attention_bias
        )
        self.q_norm = torch.nn.RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, config.rms_norm_eps)
        self.softmax_offset = torch.nn.Parameter(
            torch.empty(self.num_heads), requires_grad=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        shape = (batch, length, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden).view(shape).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        if not self.skip_rope:
            q = _rope(q, positions, self.rope_theta)
            k = _rope(k, positions, self.rope_theta)

        additive = attention_mask.expand(q.shape[0], self.num_heads, q.shape[2], -1).to(
            q.dtype
        )
        additive.masked_fill_(~attention_mask, float("-inf"))
        additive[..., -1] = self.softmax_offset.to(q.dtype).view(1, self.num_heads, 1)
        sink_shape = (q.shape[0], self.num_heads, 1, self.head_dim)
        sink_k = q.new_zeros(sink_shape)
        sink_v = q.new_zeros(sink_shape)
        k = torch.cat((k, sink_k), dim=2)
        v = torch.cat((v, sink_v), dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=additive, scale=self.scale
        )
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(out)


class MLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EagerExperts(torch.nn.Module):
    def __init__(self, config: FlintConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        hidden = config.hidden_size
        intermediate = config.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, intermediate, hidden)
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, intermediate, hidden)
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, hidden, intermediate)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, hidden_dim = hidden.shape
        flat_hidden = hidden.reshape(-1, hidden_dim)
        flat_indices = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        output = torch.zeros_like(flat_hidden)

        for expert in range(self.num_experts):
            slots = (flat_indices == expert).nonzero(as_tuple=True)[0]
            if slots.numel() == 0:
                continue
            tokens = slots // self.top_k
            x = flat_hidden[tokens]
            gate = F.linear(x, self.gate_proj[expert])
            up = F.linear(x, self.up_proj[expert])
            down = F.linear(F.silu(gate) * up, self.down_proj[expert])
            output.index_add_(0, tokens, down * flat_weights[slots].unsqueeze(-1))

        return output.view(batch, length, hidden_dim)


class SparseMoeBlock(torch.nn.Module):
    def __init__(self, config: FlintConfig):
        super().__init__()
        self.gate = torch.nn.Linear(config.hidden_size, config.num_experts, False)
        self.experts = EagerExperts(config)
        self.shared_expert = MLP(
            config.hidden_size, config.shared_expert_intermediate_size
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = self.gate(hidden).float().sigmoid()
        weights, indices = torch.topk(scores, self.experts.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden.dtype)
        return self.experts(hidden, indices, weights) + self.shared_expert(hidden)


class DecoderLayer(torch.nn.Module):
    def __init__(self, config: FlintConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        if layer_idx in set(config.mlp_only_layers or []):
            self.mlp = MLP(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(config)
        self.input_layernorm = torch.nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = torch.nn.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        del layer_idx
        hidden = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class FlintModel(torch.nn.Module):
    def __init__(self, config: FlintConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = torch.nn.RMSNorm(config.hidden_size, config.rms_norm_eps)

    @staticmethod
    def _base_mask(
        batch: int,
        length: int,
        device: torch.device,
        window: int | None,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        causal = positions[:, None] >= positions[None, :]
        if window is not None:
            causal = causal & (positions[:, None] - positions[None, :] < window)
        mask = torch.ones(batch, 1, length, length + 1, dtype=torch.bool, device=device)
        mask[..., :length] = causal
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        del use_cache
        hidden = self.embed_tokens(input_ids)
        batch, length, _ = hidden.shape
        device = hidden.device
        positions = torch.arange(length, device=device)
        layer_types = self.config.layer_types or ["full_attention"]
        full_mask = None
        swa_mask = None
        needed_full = any(t == "full_attention" for t in layer_types)
        needed_swa = any(t == "sliding_attention" for t in layer_types)
        if needed_full:
            full_mask = self._base_mask(batch, length, device, None)
        if needed_swa:
            swa_mask = self._base_mask(
                batch, length, device, self.config.sliding_window
            )

        for idx, (layer, layer_type) in enumerate(zip(self.layers, layer_types)):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                layer_idx=idx,
            )
        return self.norm(hidden)


class FlintForCausalLM(torch.nn.Module):
    def __init__(self, config: FlintConfig):
        super().__init__()
        self.config = config
        self.model = FlintModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, False)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, use_cache))

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens
