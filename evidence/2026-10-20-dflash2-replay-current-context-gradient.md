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
