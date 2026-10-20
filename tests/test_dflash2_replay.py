import torch

from mlx_gptq.dflash2_replay import (
    CONTEXT_SINK,
    CONTEXT_WINDOW,
    PromptCaches,
    _spans,
    draft_context_for,
    new_cache,
    replay_context,
    replay_fidelity,
)


class _Config:
    num_attention_heads = 2
    num_key_value_heads = 1
    hidden_size = 8


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.head_dim = 4
        self.sliding_window = 8
        self.scaling = self.head_dim**-0.5
        self.q_proj = torch.nn.Linear(8, 2 * 4, bias=False)
        self.k_proj = torch.nn.Linear(8, 1 * 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 1 * 4, bias=False)
        self.o_proj = torch.nn.Linear(2 * 4, 8, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()


class _Rope:
    def __call__(self, x, positions):
        angle = positions.float().unsqueeze(-1) * 0.01
        return torch.cos(angle), torch.sin(angle)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Identity()
        self.attention_conv = None
        self.mlp_conv = None


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])
        self.norm = torch.nn.Identity()
        self.rotary_emb = _Rope()
        self.fc = torch.nn.Linear(16, 8, bias=False)
        self.hidden_norm = torch.nn.Identity()


def test_spans_match_context_only_cache_contract():
    assert _spans(10, 0) == [(0, 10)]
    big = CONTEXT_SINK + CONTEXT_WINDOW + 5
    spans = _spans(big, 0)
    assert spans[0] == (0, CONTEXT_SINK)
    assert spans[-1][1] == big
    assert _spans(4, 100) == [(0, 4)]
    assert _spans(CONTEXT_WINDOW + 5, 100) == [(5, CONTEXT_WINDOW + 5)]


def test_prompt_caches_reset_on_out_of_order_visit():
    model = _Model()
    order = {0: [0, 1, 2]}
    caches = PromptCaches(len(model.layers), order)
    payload = {
        "context": torch.randn(1, 3, 8),
        "raw_context": None,
        "noise_embedding": torch.randn(1, 2, 8),
    }
    caches.replay(model, payload, 0, 0)
    caches.replay(model, payload, 0, 1)
    advanced = caches.state[0][1]
    assert advanced == 1
    caches.replay(model, payload, 0, 2)
    assert caches.state[0][1] == 2
    # Revisiting an earlier cycle must rebuild rather than continue.
    caches.replay(model, payload, 0, 0)
    assert caches.state[0][1] == 0


def test_replay_context_runs_and_keeps_shape():
    model = _Model()
    caches = new_cache(len(model.layers))
    context = torch.randn(1, 4, 8)
    noise = torch.randn(1, 3, 8)
    out = replay_context(model, context, noise, caches)
    assert out.shape == (1, 2, 8)
    assert caches[0]["offset"] == 4


def test_draft_context_for_uses_projection_when_shapes_match():
    model = _Model()
    payload = {"context": torch.randn(1, 3, 8), "raw_context": torch.randn(1, 3, 16)}
    projected = draft_context_for(model, payload)
    assert projected.shape == (1, 3, 8)
    payload = {"context": torch.randn(1, 3, 8), "raw_context": None}
    assert torch.equal(draft_context_for(model, payload), payload["context"])


class _FakeDataset:
    def __init__(self, payloads):
        self.payloads = payloads
        self.prompt_ids = [p["prompt_id"] for p in payloads]

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, index):
        p = self.payloads[index]
        return p, p["prompt_id"]


def test_replay_fidelity_fails_on_divergent_hidden_states():
    model = _Model()
    payloads = []
    for i in range(4):
        payloads.append({
            "prompt_id": 0,
            "context": torch.randn(1, 3, 8),
            "raw_context": None,
            "noise_embedding": torch.randn(1, 2, 8),
            "selector_hidden": torch.randn(1, 1, 8),
        })
    dataset = _FakeDataset(payloads)
    try:
        replay_fidelity(model, dataset, "cpu", limit=4, threshold=0.99)
    except RuntimeError as error:
        assert "fidelity" in str(error)
    else:
        raise AssertionError("fidelity gate must reject a divergent replay")


def test_replay_fidelity_passes_on_matching_hidden_states():
    model = _Model()
    caches = new_cache(len(model.layers))
    payloads = []
    for i in range(4):
        payload = {
            "prompt_id": 0,
            "context": torch.randn(1, 3, 8),
            "raw_context": None,
            "noise_embedding": torch.randn(1, 2, 8),
        }
        payload["selector_hidden"] = replay_context(
            model, payload["context"], payload["noise_embedding"], caches
        )
        payloads.append(payload)
    dataset = _FakeDataset(payloads)
    median, count = replay_fidelity(model, dataset, "cpu", limit=4, threshold=0.999)
    assert count == 4
    assert median > 0.999


def test_cache_is_reusable_after_no_grad_evaluation():
    """Eval under no_grad must not poison a later training backward.

    The cache persists across the initial evaluation and the first training
    step. Cached keys and values must be detached so the training graph does not
    try to backward through tensors whose graph was already freed.
    """
    model = _Model()
    order = {0: [0, 1, 2]}
    caches = PromptCaches(len(model.layers), order)

    def payload():
        return {
            "context": torch.randn(1, 3, 8),
            "raw_context": None,
            "noise_embedding": torch.randn(1, 2, 8),
        }

    # Populate the cache under no_grad, as the initial evaluation does.
    with torch.no_grad():
        caches.replay(model, payload(), 0, 0)
        caches.replay(model, payload(), 0, 1)

    # A training step reusing that cache must still be able to backward.
    hidden = caches.replay(model, payload(), 0, 2)
    hidden.sum().backward()
    assert hidden.shape == (1, 1, 8)
