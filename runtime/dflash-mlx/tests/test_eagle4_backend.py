# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from dflash_mlx.cache.snapshot import TargetHiddenChunks
from dflash_mlx.draft import eagle4_backend as eagle4_backend_module
from dflash_mlx.draft.eagle4 import (
    Eagle4DraftModel,
    Eagle4DraftModelArgs,
    quantize_eagle4_draft,
)
from dflash_mlx.draft.eagle4_backend import Eagle4DraftBackend


def _config() -> dict:
    return {
        "architectures": ["LlamaForCausalLMEagle4"],
        "attention_bias": False,
        "draft_vocab_size": 8,
        "dtype": "float32",
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
            "use_aux_hidden_state": True,
        },
        "efficient_lm_head_config": {
            "num_candidates": 2,
            "rank": 2,
            "type": "low_rank",
        },
        "fc_norm": True,
        "head_dim": 2,
        "hidden_act": "silu",
        "hidden_size": 4,
        "input_fuse_type": "concat",
        "intermediate_size": 8,
        "layer_types": ["sliding_attention"],
        "max_position_embeddings": 32,
        "mlp_bias": False,
        "model_type": "llama",
        "norm_output": True,
        "num_attention_heads": 2,
        "num_aux_hidden_states": 3,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"},
        "sliding_window": 8,
        "target_hidden_size": 4,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": 8,
    }


def _model() -> Eagle4DraftModel:
    model = Eagle4DraftModel(Eagle4DraftModelArgs.from_dict(_config()))
    mx.eval(model.parameters())
    return model


def _quantizable_model() -> Eagle4DraftModel:
    config = _config()
    config.update(
        hidden_size=32,
        target_hidden_size=32,
        head_dim=32,
        intermediate_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    model = Eagle4DraftModel(Eagle4DraftModelArgs.from_dict(config))
    mx.eval(model.parameters())
    return model


def _draft(
    model: Eagle4DraftModel,
    staged_first: int,
    root_feature: mx.array,
    *,
    block_len: int,
    suppress: list[int] | None = None,
) -> mx.array:
    suppress_mask = (
        None
        if suppress is None
        else mx.array(
            [index in suppress for index in range(model.args.vocab_size)],
            dtype=mx.bool_,
        )
    )
    return Eagle4DraftBackend().draft_greedy(
        target_model=object(),
        target_ops=object(),
        draft_model=model,
        draft_cache=[],
        staged_first=mx.array([staged_first], dtype=mx.uint32),
        draft_context=root_feature,
        block_len=block_len,
        mask_token_tail=mx.zeros((block_len - 1,), dtype=mx.uint32),
        suppress_token_mask=suppress_mask,
        async_launch=False,
    )


def test_capabilities_and_projected_body_contract():
    model = _model()
    assert model.block_size == 8
    assert model.mask_token_id == 0
    assert model.capabilities.default_block_tokens == 4
    assert model.capabilities.max_block_tokens == 16
    assert model.capabilities.supports_copyspec is False
    assert model.capabilities.supports_ddtree is False
    assert model.target_layer_ids == (1, 2, 3)

    projected = mx.random.normal((1, 3, 4), key=mx.random.key(3))
    positions = mx.array([[4, 5, 6]], dtype=mx.int32)
    hidden = model.forward_projected(mx.array([[0, 2, 7]]), projected, positions)
    raw = mx.concatenate([projected, projected, projected], axis=-1)
    expected = model.forward_projected(
        mx.array([[0, 2, 7]]),
        model.project_target_hidden(raw),
        positions,
    )
    mx.eval(hidden, expected)
    assert tuple(hidden.shape) == (1, 3, 4)
    assert bool(mx.all(mx.isfinite(hidden)))
    actual = model(mx.array([[0, 2, 7]]), raw, positions)
    mx.eval(actual)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_greedy_chain_matches_explicit_stateless_steps():
    model = _model()
    root_feature = mx.random.normal((1, 1, 4), key=mx.random.key(11))
    drafted = _draft(model, 5, root_feature, block_len=4)
    mx.eval(drafted)
    assert tuple(drafted.shape) == (3,)
    assert drafted.dtype == mx.uint32

    position_offset = int(root_feature.shape[1])
    token_rows = [mx.array([5], dtype=mx.uint32)]
    feature_rows = [root_feature[:, 0, :]]
    expected = []
    for _ in range(3):
        input_ids = mx.concatenate(token_rows)[None]
        projected = mx.concatenate(feature_rows, axis=0)[None]
        hidden = model.forward_projected(
            input_ids,
            projected,
            mx.arange(
                position_offset,
                position_offset + len(token_rows),
                dtype=mx.int32,
            )[None],
        )
        candidate_ids = model.candidate_ids(hidden[:, -1:, :]).reshape(-1)
        candidate_logits = model.candidate_logits(hidden[:, -1:, :]).reshape(-1)
        token_position = mx.argmax(candidate_logits).reshape(1).astype(mx.int32)
        token = mx.take(candidate_ids, token_position).astype(mx.uint32)
        expected.append(token)
        feature_rows.append(hidden[:, -1:, :][:, 0, :])
        token_rows.append(token)
    expected_tokens = mx.concatenate(expected, axis=0)
    mx.eval(expected_tokens)
    np.testing.assert_array_equal(np.asarray(drafted), np.asarray(expected_tokens))


def test_capture_returns_exact_candidate_topk_contract():
    model = _model()
    root_feature = mx.random.normal((1, 1, 4), key=mx.random.key(19))
    drafted, top_ids, top_logprobs = Eagle4DraftBackend().draft_greedy_capture(
        target_model=object(),
        target_ops=object(),
        draft_model=model,
        draft_cache=[],
        staged_first=mx.array([4], dtype=mx.uint32),
        draft_context=root_feature,
        block_len=3,
        mask_token_tail=mx.zeros((2,), dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
        top_width=8,
    )
    mx.eval(drafted, top_ids, top_logprobs)
    assert tuple(top_ids.shape) == (2, 2)
    assert tuple(top_logprobs.shape) == (2, 2)
    assert bool(mx.all(mx.isfinite(top_logprobs)))
    assert bool(mx.all(top_logprobs[:, 0] >= top_logprobs[:, 1]))
    for row, token in enumerate(drafted.tolist()):
        assert int(top_ids[row, 0].item()) == int(token)


def test_suppression_selects_another_exact_candidate():
    model = _model()
    root_feature = mx.random.normal((1, 1, 4), key=mx.random.key(23))
    first = _draft(model, 6, root_feature, block_len=2)
    forbidden = int(first[0].item())
    second = _draft(model, 6, root_feature, block_len=2, suppress=[forbidden])
    mx.eval(first, second)
    assert int(second[0].item()) != forbidden


def test_repeated_calls_do_not_leak_chain_state():
    model = _model()
    calls: list[mx.array] = []
    positions_seen: list[mx.array] = []

    class Recorder:
        def __getattr__(self, name):
            return getattr(model, name)

        def forward_projected(self, input_ids, projected_features, positions):
            calls.append(input_ids[0, 0].astype(mx.uint32))
            positions_seen.append(positions[0].astype(mx.int32))
            return model.forward_projected(input_ids, projected_features, positions)

    backend = Eagle4DraftBackend()
    first = backend.draft_greedy(
        target_model=object(),
        target_ops=object(),
        draft_model=Recorder(),
        draft_cache=[],
        staged_first=mx.array([7], dtype=mx.uint32),
        draft_context=mx.random.normal((1, 1, 4), key=mx.random.key(31)),
        block_len=2,
        mask_token_tail=mx.zeros((1,), dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    second = backend.draft_greedy(
        target_model=object(),
        target_ops=object(),
        draft_model=Recorder(),
        draft_cache=[],
        staged_first=mx.array([4], dtype=mx.uint32),
        draft_context=mx.random.normal((1, 2, 4), key=mx.random.key(37)),
        block_len=3,
        mask_token_tail=mx.zeros((2,), dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    mx.eval(first, second, *calls, *positions_seen)
    assert [int(call.item()) for call in calls] == [7, 4, 4]
    assert tuple(first.shape) == (1,)
    assert tuple(second.shape) == (2,)
    assert [row.tolist() for row in positions_seen] == [[1], [2], [2, 3]]


def test_chain_uses_absolute_target_positions():
    model = _model()
    context = TargetHiddenChunks(
        total_len=5,
        chunks=(mx.zeros((1, 5, 4), dtype=mx.float32),),
        spans=((0, 5),),
    )
    positions_seen: list[mx.array] = []

    class Recorder:
        def __getattr__(self, name):
            return getattr(model, name)

        def forward_projected(self, input_ids, projected_features, positions):
            positions_seen.append(positions[0].astype(mx.int32))
            return model.forward_projected(input_ids, projected_features, positions)

    drafted = Eagle4DraftBackend().draft_greedy(
        target_model=object(),
        target_ops=object(),
        draft_model=Recorder(),
        draft_cache=[],
        staged_first=mx.array([7], dtype=mx.uint32),
        draft_context=context,
        block_len=4,
        mask_token_tail=mx.zeros((3,), dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    mx.eval(drafted, *positions_seen)
    assert [row.tolist() for row in positions_seen] == [[5], [5, 6], [5, 6, 7]]


def test_greedy_path_does_not_build_capture_topk(monkeypatch):
    model = _model()

    def fail_topk(*args, **kwargs):
        raise AssertionError("greedy drafting must not build capture top-k")

    monkeypatch.setattr(eagle4_backend_module, "masked_topk_arrays", fail_topk)
    drafted = Eagle4DraftBackend().draft_greedy(
        target_model=object(),
        target_ops=object(),
        draft_model=model,
        draft_cache=[],
        staged_first=mx.array([4], dtype=mx.uint32),
        draft_context=mx.zeros((1, 1, 4)),
        block_len=3,
        mask_token_tail=mx.zeros((2,), dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )
    mx.eval(drafted)
    assert tuple(drafted.shape) == (2,)


def test_selective_draft_quantization_preserves_head_contracts():
    model = _quantizable_model()
    touched = quantize_eagle4_draft(model, bits=4, group_size=32)

    assert touched == (
        "fc",
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
    )
    assert isinstance(model.fc, nn.QuantizedLinear)
    assert isinstance(model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(model.layers[0].mlp.gate_proj, nn.QuantizedLinear)
    assert not isinstance(model.embed_tokens, nn.QuantizedEmbedding)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)
    assert not isinstance(model.efficient_lm_head, nn.QuantizedLinear)
    assert not isinstance(model.efficient_lm_head_down_proj, nn.QuantizedLinear)

    hidden = model.forward_projected(
        mx.array([[0]]),
        mx.zeros((1, 1, 32)),
        mx.array([[0]], dtype=mx.int32),
    )
    candidate_ids = model.candidate_ids(hidden)
    candidate_logits = model.candidate_logits(hidden)
    mx.eval(hidden, candidate_ids, candidate_logits)
    assert tuple(candidate_ids.shape) == (1, 1, 2)
    assert bool(mx.all(mx.isfinite(candidate_logits)))


def test_backend_rejects_non_greedy_paths():
    backend = Eagle4DraftBackend()
    with pytest.raises(ValueError, match="greedy chain drafting"):
        backend.propose_block()
    with pytest.raises(ValueError, match="DDTree"):
        backend.draft_with_topk()
    with pytest.raises(ValueError, match="branch drafting"):
        backend.draft_branch_blocks_batch()
