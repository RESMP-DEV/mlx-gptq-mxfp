# DFlash2 replay current-context gradient repair

Date: 2026-10-20
Agent: Codex

## Finding

The completed 128-cycle CPU smoke exposed one exactly-zero LoRA update: `fc.b`.
The other 45 LoRA `b` matrices moved. Inspection of the replay showed why: the
cache-safety fix detached the current cycle's projected context keys and values
before attention. That made evaluation reuse safe, but severed every gradient
through `fc`; the transformer adapters still received gradients through the
noise branch.

## Repair

`_layer_attention` now keeps the current cycle's context keys and values in the
autograd graph while storing detached copies in the ContextOnly cache. Attention
for the current step sees prior detached cache entries plus the differentiable
current context and noise entries. This is truncated backpropagation through
time: prior context contributes values without retaining an unbounded graph.

A shape assertion ties context key count to logical position count before SDPA.
Fresh-cache and prior-cache regression tests both require a nonzero `fc`
gradient. The real-cycle fidelity test remains mandatory and passes at the
0.97 threshold.

## Live diagnostic

A real captured cycle through the promoted G3 model produced:

```text
fc_a_grad_max = 0.0
fc_b_grad_max = 1.2302544117973877
```

`fc.a` is zero at initialization because `fc.b` starts at zero; the nonzero
`fc.b` gradient unblocks both parameters on subsequent optimizer steps. The
earlier all-45/46 result is therefore superseded and must not be used as final
full-head acceptance.

## Validation

```text
.venv/bin/python -m pytest tests/ -q
61 passed, 5 skipped
```

## Non-claims

- The prior 128-cycle artifact is not promoted or uploaded.
- This repair proves gradient flow and replay fidelity, not live acceptance.
- A fresh full-vocabulary smoke or GPU run is required before objective-level
  conclusions.

## Update: capture-order training

A read-only audit correctly found that the trainer's random global shuffle put
nearly every replay visit out of prompt-local capture order. `PromptCaches`
would reset at a later cycle without reconstructing its skipped prefix, so RoPE
and cache age no longer matched serving even when exported prompt groups were
contiguous.

The trainer now visits selected cycles in prompt-local capture order and rejects
`--stratified-limit` with replay enabled. The fidelity sampler now takes
contiguous prefixes across multiple prompts and cache ages instead of only the
first global slice. A retained-prefix fix also keeps differentiable current
context aligned with trimmed cache positions.

An attempted v3 launch exposed the trim/count mismatch and exited before
training; it was stopped and is not an artifact. After repair:

```text
.venv/bin/python -m pytest tests/ -q
63 passed, 5 skipped
```

This update still does not claim model quality or live acceptance.

## Mechanical run gate

`tools/validate_dflash2_run.py` checks expected step count, replay-fidelity
threshold, exactly 46 nonzero current checkpoint LoRA `b` tensors, and, for
QAT, exactly 47 prescribed code/scale pairs. It separately reports whether the
selected export moved; validation may legitimately restore the initial
parameters when its offline metric does not improve. `--require-selected-update`
turns that observation into a failure for candidate-generation runs. Validation
commands:

```text
uvx ruff check tools/validate_dflash2_run.py \
  tests/test_validate_dflash2_run.py
exit 0

.venv/bin/python -m pytest tests/test_validate_dflash2_run.py -q
3 passed
```

The superseded v2 artifact fails the checkpoint gate with `fc.b` unchanged.

## Corrected four-cycle full-vocabulary smoke

The fresh prompt-disjoint split used three training prompts/four contiguous
cycles and one validation prompt/cycle. The corrected run exited 0 with:

```text
replay_fidelity_cosine = 0.9936231970787048
steps = 4
checkpoint_b_tensors = 46
selected_b_tensors = 0
selected_initial = true
```

All 46 current checkpoint `b` tensors moved, proving the repaired gradient path.
The selected export restored the initial head because one validation cycle
became worse; that is correct selection behavior, not a failed training step.
This is a mechanism smoke, not model-quality evidence.
