# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# EAGLE-4 formula provenance: eagle-hq/EAGLE-4@f6149b0,
# LicenseRef-Eagle-4-Source-Available; internal RESMP research port.

"""Correctness-only EAGLE-4 draft support for ``LlamaForCausalLMEagle4``.

The implementation mirrors the pinned dense single-layer draft: three RMS-normed
target aux states feed ``fc``, independently normed embeddings and projected
features feed a one-layer sliding-window Llama block, and a rank-R score head
selects candidates for exact rescoring by the full draft LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP

EAGLE4_ARCHITECTURE = "LlamaForCausalLMEagle4"
EAGLE4_NUM_AUX_HIDDEN_STATES = 3
EAGLE4_LAYER_TYPES = frozenset(("sliding_attention",))

# Structural and provenance keys understood by this correctness-only parser.
# Unsupported structural choices are rejected explicitly below rather than
# silently mapped onto a different draft.
EAGLE4_CONFIG_KEYS = frozenset(
    {
        "_torchspec_version",
        "architectures",
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "draft_vocab_size",
        "dtype",
        "eagle_config",
        "efficient_lm_head_config",
        "eos_token_id",
        "fc_norm",
        "enable_fc_norm",
        "enable_post_mlp_norm",
        "head_dim",
        "initializer_range",
        "hidden_act",
        "hidden_size",
        "input_fuse_type",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "mlp_bias",
        "model_type",
        "norm_output",
        "num_attention_heads",
        "num_aux_hidden_states",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "pretraining_tp",
        "quantization_config",
        "rope_parameters",
        "rms_norm_eps",
        "share_draft_embedding_from_target",
        "sliding_window",
        "target_hidden_size",
        "tie_word_embeddings",
        "torch_dtype",
        "transformers_version",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
    }
)
_SUPPORTED_DTYPES = frozenset(("bfloat16", "float16", "float32"))
_MLX_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


def has_eagle4_architecture(config: Mapping[str, Any]) -> bool:
    return EAGLE4_ARCHITECTURE in tuple(
        str(value) for value in config.get("architectures") or ()
    )


def _require_bool(data: Mapping[str, Any], name: str, value: bool) -> None:
    if data.get(name, value) is not value:
        raise ValueError(f"EAGLE4 draft config requires {name}={value!r}")


def _positive_int(data: Mapping[str, Any], name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"EAGLE4 draft {name} must be a positive integer")
    return value


def _positive_float(data: Mapping[str, Any], name: str) -> float:
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"EAGLE4 draft {name} must be a positive number")
    return float(value)


def normalize_eagle4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the dense single-layer EAGLE-4 config contract."""
    if not isinstance(config, Mapping):
        raise ValueError("EAGLE4 draft config must be an object")
    if not has_eagle4_architecture(config):
        raise ValueError(
            f"EAGLE4 draft config must declare architectures=['{EAGLE4_ARCHITECTURE}']"
        )

    data = dict(config)
    unknown = sorted(set(data) - EAGLE4_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"Unsupported EAGLE4 draft config key(s): {', '.join(unknown)}"
        )
    if data.get("model_type") != "llama":
        raise ValueError("EAGLE4 draft config requires model_type='llama'")

    _require_bool(data, "attention_bias", False)
    _require_bool(data, "mlp_bias", False)
    _require_bool(data, "tie_word_embeddings", False)
    _require_bool(data, "fc_norm", True)
    _require_bool(data, "norm_output", True)
    _require_bool(data, "use_sliding_window", True)
    _require_bool(data, "enable_post_mlp_norm", False)
    _require_bool(data, "share_draft_embedding_from_target", True)
    if data.get("enable_fc_norm", True) is not True:
        raise ValueError("EAGLE4 draft config requires enable_fc_norm=true")
    if data.get("input_fuse_type", "concat") != "concat":
        raise ValueError("EAGLE4 draft input_fuse_type must be 'concat'")
    if data.get("hidden_act", "silu") != "silu":
        raise ValueError("EAGLE4 MLX support requires hidden_act='silu'")
    if data.get("attention_dropout", 0.0) != 0.0:
        raise ValueError("EAGLE4 MLX support requires attention_dropout=0.0")
    if data.get("quantization_config") is not None:
        raise ValueError("EAGLE4 quantized draft checkpoints are not supported")
    _positive_float(data, "rms_norm_eps")

    dtype = data.get("dtype", data.get("torch_dtype", "bfloat16"))
    torch_dtype = data.get("torch_dtype", dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"EAGLE4 draft dtype {dtype!r} is not supported")
    if torch_dtype != dtype:
        raise ValueError("EAGLE4 draft dtype and torch_dtype must agree")
    data["dtype"] = dtype

    hidden_size = _positive_int(data, "hidden_size")
    _positive_int(data, "target_hidden_size")
    _positive_int(data, "intermediate_size")
    num_attention_heads = _positive_int(data, "num_attention_heads")
    num_key_value_heads = _positive_int(data, "num_key_value_heads")
    vocab_size = _positive_int(data, "vocab_size")
    if hidden_size % num_attention_heads:
        raise ValueError("EAGLE4 hidden_size must be divisible by num_attention_heads")
    if num_attention_heads % num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    head_dim = data.get("head_dim", hidden_size // num_attention_heads)
    if head_dim != hidden_size // num_attention_heads:
        raise ValueError("EAGLE4 MLX support requires head_dim == hidden_size / heads")

    if _positive_int(data, "num_hidden_layers") != 1:
        raise ValueError("EAGLE4 MLX support requires one draft layer")
    if _positive_int(data, "num_aux_hidden_states") != EAGLE4_NUM_AUX_HIDDEN_STATES:
        raise ValueError("EAGLE4 requires exactly three aux hidden states")
    if _positive_int(data, "draft_vocab_size") != vocab_size:
        raise ValueError("EAGLE4 requires a full-vocabulary draft head")

    layer_types = tuple(data.get("layer_types") or ())
    if set(layer_types) != EAGLE4_LAYER_TYPES:
        raise ValueError(
            "EAGLE4 MLX support requires layer_types=['sliding_attention']"
        )
    sliding_window = _positive_int(data, "sliding_window")
    rope = data.get("rope_parameters")
    if not isinstance(rope, Mapping) or (
        rope.get("rope_type", rope.get("type", "default")) != "default"
    ):
        raise ValueError("EAGLE4 MLX support requires default RoPE parameters")
    rope_theta = rope.get("rope_theta")
    if (
        isinstance(rope_theta, bool)
        or not isinstance(rope_theta, (int, float))
        or rope_theta <= 0
    ):
        raise ValueError("EAGLE4 draft rope_theta must be positive")

    eagle_config = data.get("eagle_config")
    if (
        not isinstance(eagle_config, Mapping)
        or eagle_config.get("use_aux_hidden_state") is not True
    ):
        raise ValueError("EAGLE4 requires eagle_config.use_aux_hidden_state=true")
    aux_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if not isinstance(aux_ids, list) or len(aux_ids) != EAGLE4_NUM_AUX_HIDDEN_STATES:
        raise ValueError("EAGLE4 requires exactly three aux hidden-state layer IDs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in aux_ids
    ):
        raise ValueError(
            "EAGLE4 aux hidden-state layer IDs must be nonnegative integers"
        )

    head = data.get("efficient_lm_head_config")
    if not isinstance(head, Mapping):
        raise ValueError("EAGLE4 requires efficient_lm_head_config")
    if set(head) - {"type", "rank", "num_candidates", "smooth_factor"}:
        raise ValueError("EAGLE4 efficient_lm_head_config has unsupported keys")
    if head.get("type") != "low_rank":
        raise ValueError("EAGLE4 MLX support requires efficient head type='low_rank'")
    rank = _positive_int(head, "rank")
    num_candidates = _positive_int(head, "num_candidates")
    if rank > min(hidden_size, vocab_size):
        raise ValueError("EAGLE4 candidate-head rank exceeds its matrix dimensions")
    if num_candidates > vocab_size:
        raise ValueError("EAGLE4 num_candidates exceeds the vocabulary")

    data["layer_types"] = layer_types
    data["rope_scaling"] = None
    data["rope_theta"] = rope_theta
    data["sliding_window"] = sliding_window
    return data


@dataclass(frozen=True)
class Eagle4DraftModelArgs:
    model_type: str
    hidden_size: int
    target_hidden_size: int
    intermediate_size: int
    vocab_size: int
    draft_vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    dtype: str
    rope_scaling: dict[str, Any] | None
    layer_types: tuple[str, ...]
    sliding_window: int
    num_candidates: int
    candidate_rank: int
    aux_hidden_state_layer_ids: tuple[int, ...]

    @property
    def aux_hidden_width(self) -> int:
        return self.target_hidden_size * EAGLE4_NUM_AUX_HIDDEN_STATES

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Eagle4DraftModelArgs":
        data = normalize_eagle4_config(params)
        head = data["efficient_lm_head_config"]
        eagle = data["eagle_config"]
        return cls(
            model_type=str(data["model_type"]),
            hidden_size=int(data["hidden_size"]),
            target_hidden_size=int(data["target_hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            vocab_size=int(data["vocab_size"]),
            draft_vocab_size=int(data["draft_vocab_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(
                data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])
            ),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            max_position_embeddings=_positive_int(data, "max_position_embeddings"),
            rope_theta=float(data["rope_theta"]),
            dtype=str(data["dtype"]),
            rope_scaling=None,
            layer_types=tuple(data["layer_types"]),
            sliding_window=int(data["sliding_window"]),
            num_candidates=int(head["num_candidates"]),
            candidate_rank=int(head["rank"]),
            aux_hidden_state_layer_ids=tuple(eagle["eagle_aux_hidden_state_layer_ids"]),
        )

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return self.aux_hidden_state_layer_ids


def _validate_weights(
    weights: Mapping[str, mx.array], args: Eagle4DraftModelArgs
) -> None:
    expected = {
        "efficient_lm_head.weight": (args.vocab_size, args.candidate_rank),
        "efficient_lm_head_down_proj.weight": (args.candidate_rank, args.hidden_size),
        "embed_tokens.weight": (args.vocab_size, args.hidden_size),
        "fc.weight": (
            args.hidden_size,
            args.target_hidden_size * EAGLE4_NUM_AUX_HIDDEN_STATES,
        ),
        "fc_norm.0.weight": (args.target_hidden_size,),
        "fc_norm.1.weight": (args.target_hidden_size,),
        "fc_norm.2.weight": (args.target_hidden_size,),
        "layers.0.hidden_norm.weight": (args.hidden_size,),
        "layers.0.input_layernorm.weight": (args.hidden_size,),
        "layers.0.mlp.down_proj.weight": (args.hidden_size, args.intermediate_size),
        "layers.0.mlp.gate_proj.weight": (args.intermediate_size, args.hidden_size),
        "layers.0.mlp.up_proj.weight": (args.intermediate_size, args.hidden_size),
        "layers.0.post_attention_layernorm.weight": (args.hidden_size,),
        "layers.0.self_attn.k_proj.weight": (
            args.num_key_value_heads * args.head_dim,
            2 * args.hidden_size,
        ),
        "layers.0.self_attn.o_proj.weight": (
            args.hidden_size,
            args.num_attention_heads * args.head_dim,
        ),
        "layers.0.self_attn.q_proj.weight": (
            args.num_attention_heads * args.head_dim,
            2 * args.hidden_size,
        ),
        "layers.0.self_attn.v_proj.weight": (
            args.num_key_value_heads * args.head_dim,
            2 * args.hidden_size,
        ),
        "lm_head.weight": (args.vocab_size, args.hidden_size),
        "norm.weight": (args.hidden_size,),
    }
    actual = set(weights)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("Invalid EAGLE4 weights: " + ", ".join(details))

    dtypes = {weights[name].dtype for name in expected}
    expected_dtype = _MLX_DTYPES[args.dtype]
    if dtypes != {expected_dtype}:
        raise ValueError(f"EAGLE4 weights must all have configured dtype {args.dtype}")
    mismatched = [
        (name, tuple(weights[name].shape), shape)
        for name, shape in expected.items()
        if tuple(weights[name].shape) != shape
    ]
    if mismatched:
        name, actual_shape, expected_shape = mismatched[0]
        raise ValueError(
            f"EAGLE4 weight {name} has shape {actual_shape}, expected {expected_shape}"
        )


class Eagle4Attention(nn.Module):
    def __init__(self, args: Eagle4DraftModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.sliding_window = args.sliding_window
        self.q_proj = nn.Linear(
            2 * args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            2 * args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            2 * args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.rope_base = float(args.rope_theta)

    @staticmethod
    def _validate_positions(positions: mx.array, batch: int, length: int) -> None:
        if tuple(positions.shape) != (batch, length) or not mx.issubdtype(
            positions.dtype, mx.integer
        ):
            raise ValueError("EAGLE4 positions must have shape [batch, sequence]")
        if bool(mx.any(positions < 0)):
            raise ValueError("EAGLE4 positions must be nonnegative")

    def __call__(self, hidden: mx.array, positions: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        self._validate_positions(positions, batch, length)
        queries = self.q_proj(hidden).reshape(
            batch, length, self.n_heads, self.head_dim
        )
        keys = self.k_proj(hidden).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )

        row_positions = positions.reshape(batch * length)
        queries = mx.fast.rope(
            queries.reshape(batch * length, self.n_heads, self.head_dim),
            self.head_dim,
            traditional=False,
            base=self.rope_base,
            scale=1.0,
            offset=row_positions,
        ).reshape(batch, length, self.n_heads, self.head_dim)
        keys = mx.fast.rope(
            keys.reshape(batch * length, self.n_kv_heads, self.head_dim),
            self.head_dim,
            traditional=False,
            base=self.rope_base,
            scale=1.0,
            offset=row_positions,
        ).reshape(batch, length, self.n_kv_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        query_positions = positions[:, :, None]
        key_positions = positions[:, None, :]
        mask = (query_positions >= key_positions) & (
            query_positions < key_positions + self.sliding_window
        )
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class Eagle4DecoderLayer(nn.Module):
    def __init__(self, args: Eagle4DraftModelArgs):
        super().__init__()
        self.self_attn = Eagle4Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self, embedding: mx.array, projected_features: mx.array, positions: mx.array
    ) -> mx.array:
        residual = projected_features
        hidden = mx.concatenate(
            (self.input_layernorm(embedding), self.hidden_norm(projected_features)),
            axis=-1,
        )
        hidden = residual + self.self_attn(hidden, positions)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Eagle4DraftModel(nn.Module):
    """Dense EAGLE-4 drafter with full-logits and selected-candidate paths."""

    def __init__(self, args: Eagle4DraftModelArgs):
        super().__init__()
        self.args = args
        self.model_type = "eagle4"
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.fc_norm = [
            nn.RMSNorm(args.target_hidden_size, eps=args.rms_norm_eps)
            for _ in range(EAGLE4_NUM_AUX_HIDDEN_STATES)
        ]
        self.fc = nn.Linear(
            args.target_hidden_size * EAGLE4_NUM_AUX_HIDDEN_STATES,
            args.hidden_size,
            bias=False,
        )
        self.layers = [Eagle4DecoderLayer(args)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.efficient_lm_head = nn.Linear(
            args.candidate_rank, args.vocab_size, bias=False
        )
        self.efficient_lm_head_down_proj = nn.Linear(
            args.hidden_size, args.candidate_rank, bias=False
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        _validate_weights(weights, self.args)
        return weights

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        text_model = target_ops.text_model(target_model)
        target_weight = text_model.embed_tokens.weight
        expected = (self.args.vocab_size, self.args.hidden_size)
        if tuple(target_weight.shape) != expected:
            raise ValueError(
                "EAGLE-4 target embedding shape "
                f"{tuple(target_weight.shape)} does not match {expected}"
            )
        self.embed_tokens.weight = target_weight

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        width = self.args.aux_hidden_width
        if target_hidden.ndim != 3 or target_hidden.shape[-1] != width:
            raise ValueError(
                "EAGLE4 target hidden states must have shape "
                f"[batch, sequence, {width}]"
            )
        chunks = mx.split(target_hidden, EAGLE4_NUM_AUX_HIDDEN_STATES, axis=-1)
        normed = [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks, strict=True)]
        return self.fc(mx.concatenate(normed, axis=-1))

    def body(
        self,
        input_ids: mx.array,
        target_hidden: mx.array,
        positions: mx.array | None = None,
    ) -> mx.array:
        if input_ids.ndim != 2 or not mx.issubdtype(input_ids.dtype, mx.integer):
            raise ValueError(
                "EAGLE4 input_ids must have integer shape [batch, sequence]"
            )
        batch, length = input_ids.shape
        if target_hidden.shape[:2] != (batch, length):
            raise ValueError(
                "EAGLE4 input_ids and target hidden states disagree on sequence shape"
            )
        if positions is None:
            positions = mx.arange(length, dtype=mx.int32)[None, :]
            positions = mx.broadcast_to(positions, (batch, length))

        embedding = self.embed_tokens(input_ids)
        projected = self.project_target_hidden(target_hidden)
        hidden = self.layers[0](embedding, projected, positions)
        return self.norm(hidden)

    def __call__(
        self,
        input_ids: mx.array,
        target_hidden: mx.array,
        positions: mx.array | None = None,
    ) -> mx.array:
        return self.body(input_ids, target_hidden, positions)

    def full_logits(self, hidden: mx.array) -> mx.array:
        if hidden.shape[-1] != self.args.hidden_size:
            raise ValueError("EAGLE4 full-logits input must have draft hidden_size")
        return self.lm_head(hidden)

    def candidate_ids(self, hidden: mx.array) -> mx.array:
        """Select global token IDs using only rank-R approximate scores."""
        if hidden.shape[-1] != self.args.hidden_size:
            raise ValueError("EAGLE4 candidate input must have draft hidden_size")
        low_rank = hidden @ self.efficient_lm_head_down_proj.weight.T
        scores = low_rank @ self.efficient_lm_head.weight.T
        selected = mx.argpartition(scores, -self.args.num_candidates, axis=-1)[
            ..., -self.args.num_candidates :
        ]
        candidate_scores = mx.take_along_axis(scores, selected, axis=-1)
        order = mx.argsort(candidate_scores, axis=-1)[..., ::-1]
        return mx.take_along_axis(selected, order, axis=-1)

    def candidate_logits(self, hidden: mx.array) -> mx.array:
        """Return exact full-head logits for the globally selected candidates."""
        selected_ids = self.candidate_ids(hidden)
        flat_ids = selected_ids.reshape(-1)
        selected_rows = mx.take(self.lm_head.weight, flat_ids, axis=0)
        selected_rows = selected_rows.reshape((*selected_ids.shape, hidden.shape[-1]))
        return mx.matmul(selected_rows, hidden[..., None])[..., 0]
