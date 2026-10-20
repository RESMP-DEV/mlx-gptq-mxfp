# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file


import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from dflash_mlx.draft.checkpoint import get_draft_model_classes
from dflash_mlx.draft.eagle4 import (
    Eagle4DraftModel,
    Eagle4DraftModelArgs,
    normalize_eagle4_config,
)
from dflash_mlx.runtime import loading as runtime_loading
from mlx.utils import tree_flatten


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
        "sliding_window": 3,
        "target_hidden_size": 4,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": 8,
    }


def _model() -> Eagle4DraftModel:
    model = Eagle4DraftModel(Eagle4DraftModelArgs.from_dict(_config()))
    mx.eval(model.parameters())
    return model


def _rmsnorm(weight: mx.array, values: mx.array) -> mx.array:
    scale = mx.rsqrt(mx.mean(values * values, axis=-1) + 1e-6)
    return mx.expand_dims(scale, axis=-1) * values * weight


def test_checkpoint_dispatch_and_config_parsing():
    args = Eagle4DraftModelArgs.from_dict(_config())
    assert get_draft_model_classes(_config()) == (
        Eagle4DraftModel,
        Eagle4DraftModelArgs,
    )
    assert args.aux_hidden_state_layer_ids == (1, 2, 3)
    assert args.target_layer_ids == args.aux_hidden_state_layer_ids
    assert args.num_candidates == 2
    assert args.candidate_rank == 2
    assert args.sliding_window == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"architectures": ["Other"]}, "architectures"),
        ({"num_aux_hidden_states": 2}, "exactly three"),
        ({"layer_types": ["full_attention"]}, "sliding_attention"),
        ({"sliding_window": 0}, "positive integer"),
        ({"unknown_structure": True}, "Unsupported EAGLE4"),
        ({"share_draft_embedding_from_target": False}, "share_draft_embedding"),
        ({"torch_dtype": "float16"}, "dtype and torch_dtype must agree"),
    ],
)
def test_invalid_configs_rejected(mutation: dict, message: str):
    data = _config()
    data.update(mutation)
    with pytest.raises(ValueError, match=message):
        normalize_eagle4_config(data)


def test_checkpoint_metadata_keys_are_accepted():
    data = _config()
    data.update(
        {
            "_torchspec_version": "0.1.0",
            "attention_dropout": 0.0,
            "bos_token_id": 7,
            "eos_token_id": 7,
            "initializer_range": 0.02,
            "pad_token_id": 7,
            "pretraining_tp": 1,
            "torch_dtype": "float32",
            "transformers_version": "5.8.1",
            "use_cache": True,
        }
    )

    assert normalize_eagle4_config(data)["model_type"] == "llama"


def test_load_draft_bundle_loads_eagle4_checkpoint(tmp_path):
    config = _config()
    model = Eagle4DraftModel(Eagle4DraftModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded_model, meta = runtime_loading.load_draft_bundle(tmp_path, lazy=False)

    assert isinstance(loaded_model, Eagle4DraftModel)
    assert meta["config"]["architectures"] == ["LlamaForCausalLMEagle4"]


def test_load_draft_bundle_rejects_extra_eagle4_lm_head_tensor(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLMEagle4"]})
    )
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "lm_head.weight": mx.zeros((8, 4), dtype=mx.float32),
            "lm_head.bias": mx.zeros((8,), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="must contain exactly lm_head.weight"):
        runtime_loading.load_draft_bundle(tmp_path)


def test_rejects_weighted_low_rank_head():
    data = _config()
    data["efficient_lm_head_config"]["type"] = "weighted_low_rank"
    with pytest.raises(ValueError, match="type='low_rank'"):
        normalize_eagle4_config(data)


def test_aux_and_input_are_independently_normalized():
    model = _model()
    values = mx.array(
        [
            [
                [1.0, -2.0, 3.0, 4.0],
                [2.0, 2.0, -1.0, 0.5],
                [-0.25, 1.5, 2.0, -3.0],
            ]
        ]
    )
    target = mx.repeat(values, 3, axis=-1)
    expected = mx.concatenate(
        [
            _rmsnorm(norm.weight, chunk)
            for norm, chunk in zip(model.fc_norm, mx.split(target, 3, axis=-1))
        ],
        axis=-1,
    )
    projected = model.project_target_hidden(target)
    mx.eval(projected, expected)
    np.testing.assert_allclose(
        np.asarray(projected),
        np.asarray(model.fc(expected)),
        rtol=1e-6,
        atol=1e-6,
    )

    embedding = model.embed_tokens(mx.array([[0, 3, 7]]))
    layer = model.layers[0]
    normed_embedding = layer.input_layernorm(embedding)
    normed_feature = layer.hidden_norm(projected)
    assert tuple(normed_embedding.shape) == (1, 3, 4)
    assert tuple(normed_feature.shape) == (1, 3, 4)
    np.testing.assert_allclose(
        np.asarray(normed_embedding),
        np.asarray(_rmsnorm(layer.input_layernorm.weight, embedding)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(normed_feature),
        np.asarray(_rmsnorm(layer.hidden_norm.weight, projected)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_candidate_selection_and_exact_rescoring_match_full_head():
    model = _model()
    hidden = mx.array([[[0.3, -0.7, 1.1, 0.2], [-1.0, 0.5, 0.1, 0.8]]])
    mx.eval(hidden, model.parameters())

    low_rank = (
        np.asarray(hidden) @ np.asarray(model.efficient_lm_head_down_proj.weight).T
    )
    approximate = low_rank @ np.asarray(model.efficient_lm_head.weight).T
    expected_ids = np.argsort(approximate, axis=-1)[..., ::-1][..., :2]
    selected_ids = model.candidate_ids(hidden)
    np.testing.assert_array_equal(np.asarray(selected_ids), expected_ids)

    exact_reference = np.asarray(model.full_logits(hidden))
    expected_logits = np.take_along_axis(exact_reference, expected_ids, axis=-1)
    actual_logits = np.asarray(model.candidate_logits(hidden))
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)
    assert tuple(actual_logits.shape) == (1, 2, 2)


def test_strict_weight_names_shapes_and_dtypes():
    model = _model()
    weights = dict(tree_flatten(model.parameters()))
    model.sanitize(weights)

    extra = dict(weights)
    extra["unexpected.weight"] = mx.zeros((1,), dtype=mx.float32)
    with pytest.raises(ValueError, match="extra=\\['unexpected.weight'\\]"):
        model.sanitize(extra)

    bad_shape = dict(weights)
    bad_shape["fc.weight"] = mx.zeros((5, 12), dtype=mx.float32)
    with pytest.raises(ValueError, match="fc.weight has shape"):
        model.sanitize(bad_shape)

    bad_dtype = dict(weights)
    bad_dtype["norm.weight"] = mx.zeros((4,), dtype=mx.int32)
    with pytest.raises(ValueError, match="configured dtype float32"):
        model.sanitize(bad_dtype)

    mismatched_dtype = dict(weights)
    mismatched_dtype["norm.weight"] = mx.zeros((4,), dtype=mx.float16)
    with pytest.raises(ValueError, match="configured dtype float32"):
        model.sanitize(mismatched_dtype)


def test_bind_target_model_shares_target_embedding():
    model = _model()
    target_weight = mx.ones((8, 4), dtype=mx.float32)
    target_ops = SimpleNamespace(
        text_model=lambda _target: SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=target_weight)
        )
    )

    model.bind_target_model(object(), target_ops=target_ops)

    assert model.embed_tokens.weight is target_weight


def test_one_step_forward_and_candidate_path_are_finite():
    model = _model()
    input_ids = mx.array([[1, 4, 2]])
    target = mx.random.normal((1, 3, 12), key=mx.random.key(7))
    positions = mx.array([[10, 11, 12]], dtype=mx.int32)
    hidden = model(input_ids, target, positions)
    logits = model.full_logits(hidden)
    selected_logits = model.candidate_logits(hidden)
    mx.eval(hidden, logits, selected_logits)

    assert tuple(hidden.shape) == (1, 3, 4)
    assert tuple(logits.shape) == (1, 3, 8)
    assert tuple(selected_logits.shape) == (1, 3, 2)
    assert bool(mx.all(mx.isfinite(hidden)))
    assert bool(mx.all(mx.isfinite(logits)))
    assert bool(mx.all(mx.isfinite(selected_logits)))
