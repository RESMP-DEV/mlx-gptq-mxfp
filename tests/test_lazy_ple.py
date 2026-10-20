import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from mlx_gptq.sequential import LazyShardedEmbedding, RawShardIndex


def test_lazy_sharded_embedding_reads_selected_rows(tmp_path: Path):
    shard_0 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shard_1 = shard_0 + 100
    tensor_name = "layer.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
    save_file(
        {
            tensor_name.format(i=0): shard_0,
            tensor_name.format(i=1): shard_1,
        },
        tmp_path / "model.safetensors",
    )
    weight_map = {tensor_name.format(i=i): "model.safetensors" for i in range(2)}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    index = RawShardIndex(tmp_path)
    embedding = LazyShardedEmbedding(
        "layer.ple.ple_embedding.ngram_embedding",
        index,
        2,
        4,
        3,
        torch.device("cpu"),
        torch.float32,
    )
    ids = torch.tensor([[0, 4, 7], [5, 1, 3]])
    expected = torch.cat([shard_0, shard_1])[ids.reshape(-1)].reshape(2, 3, 3)

    assert torch.equal(embedding(ids), expected)
