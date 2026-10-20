from types import SimpleNamespace

import torch

from mlx_gptq import sequential
from mlx_gptq.sequential import Pipeline, materialize_module


class ContextLayer(torch.nn.Module):
    def __init__(self, layer_idx, observations):
        super().__init__()
        self.layer_idx = layer_idx
        self.observations = observations

    def forward(self, hidden_states, per_layer_input, shared_state, layer_tag):
        self.observations.append(
            (self.layer_idx, per_layer_input.clone(), layer_tag, dict(shared_state))
        )
        shared_state[self.layer_idx] = layer_tag
        return hidden_states + per_layer_input


class ContextModel(torch.nn.Module):
    def __init__(self, observations):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 4)
        self.layers = torch.nn.ModuleList(
            [ContextLayer(0, observations), ContextLayer(1, observations)]
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        shared_state = {}
        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = torch.full_like(hidden_states, layer_idx + 1)
            hidden_states = layer(
                hidden_states,
                per_layer_input,
                shared_state=shared_state,
                layer_tag=f"layer-{layer_idx}",
            )
        return hidden_states


class FakeShards:
    def __init__(self, tensors):
        self.tensors = tensors

    def has(self, name):
        return name in self.tensors

    def get(self, name):
        return self.tensors[name]


def test_materialize_module_loads_non_meta_checkpoint_buffers():
    module = torch.nn.Linear(2, 2, bias=False, device="meta")
    module.register_buffer("layer_scalar", torch.ones(1))
    shards = FakeShards(
        {
            "layer.weight": torch.eye(2),
            "layer.layer_scalar": torch.tensor([3.0]),
        }
    )

    materialize_module(module, "layer", shards, "cpu", torch.bfloat16)

    assert module.weight.device.type == "cpu"
    assert module.weight.dtype == torch.bfloat16
    assert module.layer_scalar.item() == 3.0


def test_capture_preserves_positional_and_per_layer_context(monkeypatch):
    observations = []
    model = ContextModel(observations)
    args = SimpleNamespace(
        devices=["cpu"],
        dtype="bfloat16",
        layers_attr="layers",
        checkpoint_model_prefix=None,
        artifact_layers_prefix=None,
    )
    pipeline = Pipeline(model, shards=None, args=args)
    monkeypatch.setattr(sequential, "materialize_module", lambda *args, **kwargs: None)
    monkeypatch.setattr(sequential, "free_module", lambda *args, **kwargs: None)

    batches = pipeline.catch_layer0_inputs(torch.tensor([[1, 2, 3]]), batch_size=1)

    assert observations == []
    assert len(batches) == 1
    hidden_states, contexts = batches[0]
    assert len(contexts) == 2
    assert torch.equal(contexts[0][0][0], torch.ones_like(hidden_states))
    assert torch.equal(contexts[1][0][0], torch.full_like(hidden_states, 2))
    assert contexts[0][1]["layer_tag"] == "layer-0"
    assert contexts[1][1]["layer_tag"] == "layer-1"
    assert contexts[0][1]["shared_state"] is contexts[1][1]["shared_state"]

    for layer_idx, layer in enumerate(model.layers):
        outputs = pipeline.forward_layer(layer, batches, layer_idx)
        batches[0] = (outputs[0], contexts)

    assert [item[2] for item in observations] == ["layer-0", "layer-1"]
    assert observations[0][3] == {}
    assert observations[1][3] == {0: "layer-0"}
    assert torch.allclose(
        batches[0][0].float(), hidden_states.float() + 3, atol=0.02, rtol=0
    )
