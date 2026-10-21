from types import SimpleNamespace

import torch
from mlx_gptq.capture import CaptureManager
from mlx_gptq.flint_torch import FlintConfig, FlintForCausalLM
from mlx_gptq.sequential import Pipeline, materialize_module


def tiny_config():
    return FlintConfig(
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        num_experts=4,
        num_experts_per_tok=1,
        sliding_window=5,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        no_rope_layer_indices=[0, 3],
        mlp_only_layers=[0],
    )


def test_flint_eager_forward_and_capture():
    torch.manual_seed(7)
    model = FlintForCausalLM(tiny_config()).to(torch.bfloat16).eval()
    inputs = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (1, 8, 128)

    layer = model.model.layers[1]
    manager = CaptureManager(layer, r"(^|\.)gate($|\.)", 32)
    assert len(manager.fused_tasks) == 3
    assert len(manager.linear_tasks) > 0
    hidden = torch.randn(1, 8, 64, dtype=torch.bfloat16)
    with manager:
        with torch.no_grad():
            layer(
                hidden,
                attention_mask=_causal_mask(hidden),
                position_ids=torch.arange(8),
                layer_idx=1,
            )
    assert any(stash.tokens > 0 for stash in manager.stashes.values())


def _causal_mask(hidden):
    length = hidden.shape[1]
    causal = torch.ones(1, 1, length, length + 1, dtype=torch.bool)
    causal[..., :length] = (
        torch.arange(length)[:, None] >= torch.arange(length)[None, :]
    )
    return causal


def test_flint_layer_materializes_from_checkpoint_names():
    config = tiny_config()
    layer = FlintForCausalLM(config).model.layers[1].to_empty(device="meta")
    tensors = {}
    prefix = "model.layers.1"
    for name, parameter in layer.named_parameters():
        if parameter.ndim == 3 and name.startswith("mlp.experts."):
            for expert in range(config.num_experts):
                suffix = name.removeprefix("mlp.experts.")
                key = f"{prefix}.mlp.experts.{expert}.{suffix}.weight"
                tensors[key] = torch.randn(parameter.shape[1:])
            continue
        tensors[f"{prefix}.{name}"] = torch.randn(parameter.shape)

    class Shards:
        def has(self, name):
            return name in tensors

        def get(self, name):
            return tensors[name]

    materialize_module(layer, prefix, Shards(), "cpu", torch.bfloat16)
    assert layer.mlp.experts.gate_proj.shape == (
        config.num_experts,
        config.moe_intermediate_size,
        config.hidden_size,
    )
    assert layer.self_attn.softmax_offset.device.type == "cpu"


def test_flint_pipeline_captures_layer_context():
    torch.manual_seed(8)
    model = FlintForCausalLM(tiny_config()).to(torch.bfloat16).eval()

    class Shards:
        def has(self, name):
            return name == "model.embed_tokens.weight"

        def get(self, name):
            assert name == "model.embed_tokens.weight"
            return model.model.embed_tokens.weight

    args = SimpleNamespace(
        devices=["cpu"],
        dtype="bfloat16",
        layers_attr="model.layers",
        checkpoint_model_prefix=None,
        artifact_layers_prefix=None,
    )
    pipeline = Pipeline(model, Shards(), args)
    batches = pipeline.catch_layer0_inputs(torch.tensor([[1, 2, 3]]), 1)
    assert batches[0][0].shape == (1, 3, 64)
    assert len(batches[0][1]) == 4
    assert "attention_mask" in batches[0][1][0][1]

    output = pipeline.forward_layer(model.model.layers[0], batches, 0)
    assert output[0].shape == (1, 3, 64)
