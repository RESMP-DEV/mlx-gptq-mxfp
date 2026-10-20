from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps


class _Embedding:
    def __call__(self, input_ids: mx.array) -> mx.array:
        return input_ids.astype(mx.float32)[..., None]


class _BoundaryLayer:
    def __init__(self, layer_index: int):
        self.layer_index = layer_index

    def __call__(
        self,
        hidden_states: mx.array,
        mask=None,
        cache=None,
    ) -> mx.array:
        return hidden_states + (self.layer_index + 1)


class _BoundaryTarget:
    def __init__(self, num_layers: int = 64):
        self.embed_tokens = _Embedding()
        self.layers = [_BoundaryLayer(index) for index in range(num_layers)]
        self.norm = lambda value: value
        self.lm_head = lambda value: value[..., :1]


def _boundary_wrapper() -> SimpleNamespace:
    model = _BoundaryTarget()
    return SimpleNamespace(
        model=model,
        lm_head=model.lm_head,
        args=SimpleNamespace(tie_word_embeddings=False),
    )


def test_eagle4_capture_ids_select_post_layer_outputs() -> None:
    target = _boundary_wrapper()
    input_ids = mx.array([[10, 11, 12]], dtype=mx.uint32)

    logits, captured = QwenGdnTargetOps().forward_with_hidden_capture(
        target,
        input_ids=input_ids,
        capture_layer_ids={4, 32, 60},
    )

    assert set(captured) == {4, 32, 60}
    for layer_id in (3, 31, 59):
        expected = input_ids.astype(mx.float32)[..., None] + sum(
            range(layer_id + 2)
        )
        actual = captured[layer_id + 1]
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected)
    mx.eval(logits)
    assert logits.shape == (1, 3, 1)


def test_eagle4_context_feature_preserves_layer_order() -> None:
    target = _boundary_wrapper()
    input_ids = mx.array([[7]], dtype=mx.uint32)
    _, captured = QwenGdnTargetOps().forward_with_hidden_capture(
        target,
        input_ids=input_ids,
        capture_layer_ids={4, 32, 60},
    )

    feature = QwenGdnTargetOps().extract_context_feature(
        captured,
        target_layer_ids=[3, 31, 59],
    )
    expected = mx.concatenate(
        [captured[4], captured[32], captured[60]],
        axis=-1,
    )
    mx.eval(feature, expected)
    assert mx.array_equal(feature, expected)
