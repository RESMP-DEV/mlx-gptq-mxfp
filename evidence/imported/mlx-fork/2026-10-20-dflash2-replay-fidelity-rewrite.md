# DFlash2 trainer replay rewrite and fidelity gate

Date: 2026-10-20
Agent: Codex

## Scope

Replace the DFlash2 distillation trainer's draft replay with a faithful
port of the runtime `ContextOnlyDraftKVCache` contract, add a startup fidelity
gate that fails any run whose replay does not match the deployed draft, pin the
behaviour with tests including real captured cycles, and document the class of
foot guns that produced the 2026-10-19 invalid QAT and full-logit results.

## What changed

### New module `mlx_gptq/dflash2_replay.py`

- `replay_context(model, context, noise, caches)` reproduces the runtime draft
  forward: per-layer context-only cache, 64-token sink plus 1024-token window
  trimming, absolute logical positions for RoPE and the sliding-window mask,
  span selection for long contexts, and grouped-query head expansion.
- `draft_context_for(model, payload)` returns the already-projected
  `hidden_norm(fc(raw_context))` that the runtime passes to the draft.
- `PromptCaches` keeps a per-prompt cache and rebuilds it whenever a prompt's
  cycles are visited out of capture order, which stratified sampling and sorted
  validation both do.
- `replay_fidelity(model, dataset, device, limit, threshold)` measures the
  median cosine between replayed hidden states and the runtime-captured
  `selector_hidden`, and raises unless it clears the threshold.

### Trainer wiring (`mlx_gptq/dflash2_distill.py`)

- `_forward` now either uses the captured `selector_hidden` shortcut or the
  cached replay; the hand-rolled per-block forward is gone.
- `--replay-draft` continues to select the cached replay for selector-mode
  training.
- New `--fidelity-threshold` (default 0.97) and `--fidelity-cycles`
  (default 32). The gate runs once before the first optimizer step, prints
  `replay_fidelity_cosine`, records it in the run receipt, and raises on
  failure.
- `replay_draft`, `qat_mxfp4`, and `fidelity_threshold` are part of the resume
  fingerprint.

## Verification

| Check | Result |
|---|---|
| Local suite `mlx-gptq-mxfp` | 53 passed, 5 skipped |
| Fidelity gate on 8 real cycles, G3 model | median cosine 0.9946, passes |
| Fidelity gate with the naive per-block replay re-introduced | median cosine 0.7645, correctly rejected |
| `test_replay_fidelity_passes_on_matching_hidden_states` | passes |
| `test_replay_fidelity_fails_on_divergent_hidden_states` | passes |
| `test_cached_replay_matches_runtime_hidden_states_on_real_cycles` | passes, skipped when capture absent |
| MXFP4 pack transfer, MLX dequantizes CUDA-side packed words and E8M0 bytes | bitwise identical, max abs diff 0.0 |

The negative control is the important one: the gate rejects the exact replay
path that invalidated the prior results, on real data, not on a synthetic
fixture.

## MXFP4 portability finding

A CUDA-trained, CUDA-quantized artifact transfers to MLX unchanged. MLX
dequantizes our packed `uint32` words and E8M0 `uint8` scale bytes
bitwise-identically, so training may run on CUDA while serving runs on MLX. The
block-scale *selection rule* differs (MLX 0.32.2 and later round the scale up;
`mlx_gptq/grid.py` rounds to nearest), but that is a policy choice within a
portable format, not a compatibility boundary. The prequantized load path never
re-runs `mx.quantize`, so a stored artifact always uses its stored scales. This
downgrades the scale-grid divergence from a correctness bug to a documented
wash, with the parity test pinning whichever rule the repo intends.

## Foot guns recorded

`RESMP-DEV/dflash-mlx` `docs/dflash2-distillation.md` gained a "Foot guns:
training-vs-runtime fidelity" section covering the cache contract, the
projected-context requirement, the selector-only gradient severance, the
fidelity gate usage, the MXFP4 grid policy, and the meta-rule that no
distillation or QAT result in this repo is meaningful until the fidelity gate
has passed.

## Non-claims

- This work does not re-run QAT, the full-logit objective, or any live gate. The
  corrected replay makes those experiments valid to run, not yet run.
- Cosine 0.9946 is not bitwise parity with the MLX runtime; the residual is
  consistent with BF16 accumulation-order differences between the MLX and
  PyTorch implementations.
- The fidelity gate uses cosine against the runtime-captured hidden states; a
  replay that is wrong in a way that preserves cosine would not be caught.
- The MXFP4 grid rule was left as-is. Regenerating any RTN or MXFP4 control
  against MLX-native round-up semantics is a separate decision that should be
  made deliberately before the next control is built.

## Commits

- `RESMP-DEV/mlx-gptq-mxfp` branch `work/dflash2-calibration`:
  `478031d` replay rewrite and gate, `216656e` real-cycle fidelity test.
- `RESMP-DEV/dflash-mlx` branch `work/local-mxfp4-dflash2`:
  `16f189d` foot-gun documentation.
