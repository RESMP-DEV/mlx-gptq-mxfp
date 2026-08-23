from types import SimpleNamespace

import torch

from mlx_gptq.sequential import Pipeline


class TextBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])


class TextOnlyRuntime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TextBackbone()


def test_multimodal_checkpoint_prefixes_are_independent():
    args = SimpleNamespace(
        devices=["cpu"],
        dtype="bfloat16",
        layers_attr="model.layers",
        checkpoint_model_prefix="model.language_model",
        artifact_layers_prefix="language_model.model.layers",
    )
    pipeline = Pipeline(TextOnlyRuntime(), shards=None, args=args)

    assert pipeline._checkpoint_name("model.embed_tokens") == (
        "model.language_model.embed_tokens"
    )
    assert pipeline.checkpoint_layers_prefix_fmt.format(7) == (
        "model.language_model.layers.7"
    )
    assert pipeline.artifact_layers_prefix_fmt.format(7) == (
        "language_model.model.layers.7"
    )
