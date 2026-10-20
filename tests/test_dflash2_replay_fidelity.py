"""Pin the cached draft replay against real runtime-captured hidden states.

This is the guard that would have caught the 2026-10-20 replay defect: the
naive per-block replay reproduced runtime hidden states at cosine ~0.73 while
the ContextOnly cache replay matches at ~0.99. It runs against the real
capture when it is present and is skipped otherwise, so it never fabricates a
pass from synthetic data.
"""

from pathlib import Path

import pytest
import torch

from mlx_gptq.dflash2_distill import CycleDataset
from mlx_gptq.dflash2_replay import replay_fidelity

CAPTURE = Path("/Users/kearm/mlx/build/data/fable-coding-896-full-logits-capture")
MODEL = Path(
    "/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-G3-g20.0625-g30.1875"
)
PACKAGE = "/Users/kearm/mlx/.worktrees/dflash-pytorch"


@pytest.mark.skipif(
    not (CAPTURE / "index.json").is_file() or not (MODEL / "config.json").is_file(),
    reason="real capture or promoted G3 model not present",
)
def test_cached_replay_matches_runtime_hidden_states_on_real_cycles():
    from mlx_gptq.dflash2_calibrate import _load_reference

    model = _load_reference(str(MODEL), PACKAGE, "cpu")
    dataset = CycleDataset(CAPTURE, limit=16, stratified=False, seed=43)
    median, count = replay_fidelity(model, dataset, "cpu", limit=16, threshold=0.97)
    assert count > 0
    assert median >= 0.97
