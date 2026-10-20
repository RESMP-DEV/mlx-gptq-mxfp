# Correction: DFlash2 QAT rejection attribution and two confirmed root causes

Date: 2026-10-20
Agent: Codex

## Correction to prior receipts

`evidence/2026-10-19-dflash2-qat-mxfp4-lane.md` concluded that QAT "does not
improve the RTN MXFP4 control" and that MXFP4 quantization itself costs live
acceptance. That attribution is wrong. The live-gate numbers in those receipts
are accurate measurements of the artifacts produced, but the conclusion about
the objective does not hold, because both the training objective and the
quantization grid were operating on incorrect implementations.

This receipt records two confirmed implementation defects that explain the
anomalous result, and supersedes the causal claims in
`evidence/2026-10-20-dflash2-qat-comparison-correction.md`.

## Defect 1: the training forward pass is not the runtime forward pass

Measured on real cycles from the completed 896-prompt full-logit capture with
the promoted G3 head, comparing reconstructed hidden states against the
runtime-captured `selector_hidden`:

| Replay variant | median cosine | median relative error |
|---|---:|---:|
| Trainer replay (no cache, positions from zero) | 0.764 | 0.790 |
| ContextOnly cache replay, projected context | 0.9946 | 0.109 |
| ContextOnly cache replay, captured context | 0.9946 | 0.109 |

Two concrete causes:

1. **Missing draft context cache.** The MLX runtime keeps a
   `ContextOnlyDraftKVCache` per layer that stores only target-context keys,
   with a 64-token sink, a 1024-token window, and an `offset` that advances by
   the context length. RoPE therefore uses absolute logical positions and the
   sliding-window mask is built from absolute key positions. The trainer
   created no cache and started positions at zero every cycle.

2. **Target input projection applied at the wrong point.** The runtime passes
   `draft_context`, which is already `hidden_norm(fc(raw_context))`. The trainer
   passed `raw_context` and applied the projection inside the layer loop, so
   `fc` was bypassed. This is also why the `fc` adapter could never learn.

Both defects were also found independently by the read-only implementation
audit worker, which ranked them above the quantization-theory explanations.

## Defect 2: the MXFP4 scale grid does not match MLX 0.32.4

`mlx_gptq/grid.py::choose_mxfp_scale` rounds the block exponent to nearest
(`torch.round(torch.log2(...))`) and documents parity against `mlx.core 0.31`.
MLX 0.32.2 changed the kernel to round the block scale **up** to avoid
saturation. Measured on the installed MLX 0.32.4 for a random weight block:

- MLX native scale bytes: `[119, 120, 120, 125, 125, 125, 126, 126]`
- `mlx_gptq` scale bytes: `[119, 119, 119, 125, 125, 125, 126, 125]`

An independent audit measured 46.9 percent of groups receiving a 2x larger
scale under the MLX round-up rule on a larger sample. The packed artifact was
unaffected because `dflash2_pack.py` injects pre-computed codes and scale bytes
and the runtime never re-runs `mx.quantize` for a prequantized draft, which is
why the earlier "export equals training weights" check reported exactly 0.0
error. The defect is that the grid itself is not MLX-native, so any RTN
baseline, parity check, or regenerated calibration performed with this grid
disagrees with the runtime.

The repository test `tests/test_grid_and_packing.py::test_mxfp_rtn_matches_mlx`
asserts exact native equality and passes only against the repo's pinned
`mlx 0.32.0`; it will fail against the installed 0.32.4.

## Why this invalidates the earlier interpretation

Every QAT and full-logit run trained through defect 1, so the objective was
evaluated on a function that differs from the deployed draft by a margin the
acceptance metric is sensitive to. The RTN control in those receipts was built
through defect 2, so the control was not MLX-native MXFP4 either. The comparison
"QAT loses to RTN" is therefore a comparison between two non-representative
artifacts and cannot support a claim about QAT.

Per the recorded literature evidence, several quantization-theory explanations
that the audit surfaced (STE gradient error, oscillating block scales, rank
versus per-group degree-of-freedom imbalance) are either not applicable or
secondary to these two defects, because this implementation recomputes scales
from a drifting maximum rather than learning them and freezes nothing.

## Required before any new experiment

1. Rewrite the trainer replay to the ContextOnly cache contract, including
   absolute positions, span selection, sink plus window trimming, and the
   already-projected `draft_context`.
2. Add a fidelity gate that fails any run whose replay cosine against captured
   runtime hidden states is below a threshold on a held-out sample.
3. Correct `choose_mxfp_scale` to match MLX 0.32.2 and later round-up semantics,
   update the stale contract comment, and regenerate any RTN or MXFP4 baseline
   used as a control.
4. Only then re-run QAT, the full-logit objective, and the live gate.

## Non-claims

- This receipt does not yet re-run any QAT or distillation experiment.
- Cosine 0.9946 is not bitwise parity with the MLX runtime; the residual is
  consistent with BF16 accumulation-order differences between the MLX and
  PyTorch implementations.
- The 46.9 percent figure is the audit worker's measurement on one random
  weight sample, reproduced here qualitatively on a smaller sample; it is a
  scale of divergence, not a final audit number.
