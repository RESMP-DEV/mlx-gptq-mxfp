# DFlash2 QAT root cause: the training forward pass did not match runtime

Date: 2026-10-20
Agent: Codex

## Summary

Every DFlash2 QAT and full-logit distillation run in this campaign optimized an
incorrect forward pass. The trainer's manual draft replay does not reproduce the
runtime draft forward, so gradients were computed against the wrong function.
This is the root cause of the anomalous "QAT is worse than RTN" result and of the
disconnected behavior of the full-logit and selector objectives.

## Measured evidence

Reconstructed hidden states were compared against the runtime-captured
`selector_hidden` on real cycles from the completed 896-prompt full-logit
capture, using the promoted G3 head.

| Replay variant | median cosine | median relative error |
|---|---:|---:|
| Trainer replay (no cache, positions from 0) | 0.764 | 0.790 |
| ContextOnly cache replay, projected context | 0.9946 | 0.109 |
| ContextOnly cache replay, captured context | 0.9946 | 0.109 |

Earlier sampling over 128 cycles of the B550 capture gave median cosine 0.580 and
median relative error 0.958 for the trainer replay. The corrected replay is
consistently above cosine 0.99.

## Defect 1: the runtime maintains a draft context cache

MLX runtime (`dflash_mlx/model.py`):

- `ContextOnlyDraftKVCache` stores only target-context keys and values, never
  speculative noise keys.
- `sink_size=64`, `window_size=1024`, applied after every append.
- `offset` advances by the context length each cycle, so RoPE uses absolute
  logical positions, not positions starting at zero.
- `_rope_context_segments` applies RoPE to each cached context span at
  `cache_offset + span_start`.
- `_attention_mask` builds the sliding-window mask from absolute key positions.

Trainer replay (`mlx_gptq/dflash2_distill.py::_forward`):

- creates no cache;
- starts `position_ids` at zero for every cycle;
- passes positions and cache through a path that does not reproduce the runtime
  context-only cache semantics.

Consequence: the draft transformer attends to a different key set at different
positions than it does while serving.

## Defect 2: the trainer never applies the target input projection

The MLX runtime calls the draft with `draft_context`, which is already the
output of `project_target_hidden` (`hidden_norm(fc(raw_context))`). The capture
stores that projected tensor in `context` and the unprojected features in
`raw_context`.

The trainer's replay path passes the unprojected `raw_context` into the layers and
applies the projection inside the loop body instead. This is both numerically
different from runtime and the reason `fc` never receives gradients: the
projection is bypassed entirely, so the `fc` adapter stays at exactly zero.

## Defect 3 (earlier, already fixed)

Selector-mode replay returned the captured `selector_hidden` directly, so no
gradient reached any draft weight. Fixed by `--replay-draft` in
`RESMP-DEV/mlx-gptq-mxfp` commit `2e92b47`, which made all 46 LoRA `b` matrices
nonzero. That fix was necessary but not sufficient: it replaced one wrong
forward pass with a different wrong forward pass.

## Consequence for recorded results

The following are invalid as evidence about the objectives themselves and remain
valid only as evidence about the training path:

- `evidence/2026-10-19-dflash2-qat-mxfp4-lane.md` (offline QAT comparisons)
- `evidence/2026-10-19-dflash2-full-logit-17500-arms.md`
- `evidence/2026-10-19-dflash2-full-logit-live-gate-decision.md`
- `evidence/2026-10-20-dflash2-qat-comparison-correction.md`

Their live-gate numbers remain accurate measurements of the artifacts those runs
produced. Their interpretation as "this objective is worse than that objective"
does not hold, because the objective was not evaluated on the deployed function.

## Next step

Rewrite the trainer replay to the ContextOnly cache contract and add a fidelity
gate that fails the run unless replay cosine against captured runtime hidden
states exceeds a threshold on a held-out sample. No further QAT or full-logit
experiment should be run before that gate passes.

## Non-claims

- This receipt does not yet re-run any QAT or distillation experiment.
- Cosine 0.9946 is not exact parity; residual error is consistent with BF16
  accumulation order between the MLX and PyTorch implementations.
- The literature sub-agents' findings are recorded separately and are not merged
  into this receipt.
