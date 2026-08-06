"""Capture: fused-expert F.linear interception and nn.Linear input dedup."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlx_gptq.capture import CaptureManager


class FusedExperts(nn.Module):
    """Mimics the transformers >= 5 fused expert layout."""

    def __init__(self, E=4, H=32, I=32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, I))
        self.num_experts = E

    def forward(self, x, top_k_index):
        out = torch.zeros_like(x)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            _, tok = torch.where(mask[e])
            if tok.numel() == 0:
                continue
            cur = x[tok]
            gate, up = F.linear(cur, self.gate_up_proj[torch.tensor(e)]).chunk(2, -1)
            h = F.silu(gate) * up
            out.index_add_(0, tok, F.linear(h, self.down_proj[torch.tensor(e)]))
        return out


class ToyLayer(nn.Module):
    def __init__(self, H=32):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.gate = nn.Linear(H, 4, bias=False)  # router: must be excluded
        self.experts = FusedExperts(H=H)

    def forward(self, x):
        y = self.q_proj(x) + self.k_proj(x)
        logits = self.gate(y)
        top = logits.topk(2, dim=-1).indices
        return self.experts(y, top)


def test_expert_routing_capture_and_dedup():
    torch.manual_seed(0)
    layer = ToyLayer()
    mgr = CaptureManager(layer, exclude_regex=r"(^|\.)gate$", group_size=32)

    # router excluded, q/k included
    names = [t.name for t in mgr.linear_tasks]
    assert "gate" not in names and set(names) == {"q_proj", "k_proj"}
    assert len(mgr.fused_tasks) == 2  # gate_up + down

    T = 64
    x = torch.randn(T, 32)
    with mgr:
        mgr.begin_batch()
        layer(x)

    # q and k saw the SAME tensor -> deduped into one stash
    keys = {t.stash_key for t in mgr.linear_tasks}
    assert len(keys) == 1
    assert mgr.stashes[keys.pop()].tokens == T

    # every token goes to exactly top-2 experts
    gu = next(t for t in mgr.fused_tasks if t.param_name == "gate_up_proj")
    dn = next(t for t in mgr.fused_tasks if t.param_name == "down_proj")
    assert sum(mgr.stashes[k].tokens for k in gu.stash_keys) == 2 * T
    assert sum(mgr.stashes[k].tokens for k in dn.stash_keys) == 2 * T
    # per-expert token counts must match between gate_up and down
    for kg, kd in zip(gu.stash_keys, dn.stash_keys):
        assert mgr.stashes[kg].tokens == mgr.stashes[kd].tokens

    # Hessian shape from stash
    H = mgr.stashes[gu.stash_keys[0]].hessian("cpu")
    assert H.shape == (32, 32) and torch.isfinite(H).all()
    mgr.check_capture_happened()
