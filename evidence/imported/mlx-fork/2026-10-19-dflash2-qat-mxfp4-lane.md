# DFlash2 QAT-style MXFP4 logit distillation: lane status and first execution

Date: 2026-10-19
Agent: Codex (B550 2x RTX 3090 Ti for training, M4 Max for MLX load and live gate)

## Direct answer to "have we done work on QAT-style logit distillation?"

Partly, and the honest state before this session was: the machinery existed, the
full-logit objective existed, but they had never been run together and never
gated. Three separate gaps existed, and this session closed them:

1. The QAT export path was silently wrong. `_merge_lora` merged the
   *unquantized* adapter product, so the exported weights did not match the
   fake-quantized weights the model was trained and validated with. Fixed in
   `RESMP-DEV/mlx-gptq-mxfp` commit `5f47153` with a regression test
   (`test_qat_merge_exports_fake_quantized_weights`).
2. No QAT run had ever been executed with the current trainer. The only prior
   artifact was a 103-step B550 `qat-smoke` from the old trainer.
3. The MXFP4 draft artifact had never been live-gated at all, so the quantized
   control was unmeasured.

## What was run

### QAT training (B550, RTX 3090 Ti, selector objective, `fable128-capture`)

Base: promoted G3 head (`c90e0fc6...`). Rank-8 LoRA, alpha 16, predicted
predecessors, 2 epochs, 18,666 steps per arm, seed 43, validation fraction 0.2.

| Arm | QAT | LR | init prefix | best prefix | init top1 | best top1 | best loss | s/cycle |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qat-lr1e-6 | yes | 1e-6 | 0.75902 | 0.76782 | 0.83209 | 0.84730 | 0.58468 | 0.0060 |
| qat-lr2e-6 | yes | 2e-6 | 0.75902 | 0.77511 | 0.83209 | 0.85493 | 0.55036 | 0.0060 |
| qat-lr4e-6 | yes | 4e-6 | 0.75902 | 0.77424 | 0.83209 | 0.85701 | 0.52057 | 0.0060 |
| bf16-lr2e-6 (control) | no | 2e-6 | 0.77821 | 0.77821 | 0.85304 | 0.85304 | 0.58014 | 0.0060 |

Reading: the MXFP4 fake-quant costs about 2 points of selector prefix at
initialization (0.759 vs 0.778) and about 2 points of top1. QAT recovers most
of that loss: at LR 2e-6 it reaches prefix 0.7751 and top1 0.8549, slightly
above the BF16 control's starting top1 of 0.85304, while still 0.3 points below
the BF16 control's prefix. The BF16 control made no progress at all at LR 2e-6
(best equals initial on every metric), so the offline comparison does not yet
show QAT beating BF16; it shows QAT closing the quantization gap.

Peak training memory 8.01 GB reserved on a 24 GB card; 0.0060 s/cycle at batch
size 1, so a 18,666-step arm is about 111 seconds. This lane is cheap.

### Export and packing correctness (verified numerically, not assumed)

- QAT training weights are fake-quantized with `choose_mxfp_scale` then
  `quantize_mxfp_with_scale`; the export must reproduce that exact grid.
- Verified on the packed artifact: codes match the RTN codes exactly, scales
  match byte for byte, and the dequantized exported weights equal the
  QAT-effective weights with `max_abs` error `0.0e+00` on `fc`,
  `layers.0.self_attn.q_proj`, `layers.3.mlp.down_proj`, and
  `candidate_selector.hidden_projection`.
- An earlier 17% discrepancy was a comparison error in the verification script
  (it compared the export against the unquantized merged weights instead of the
  QAT-effective ones). Corrected; the export path is sound.
- Artifact: `dflash-qat-mxfp4-lr2e-6` (model SHA-256
  `3b2b56f4125d532ca4aa2a03df7e0cab9afad6766743be88c65effaf4dbf6ba6`), 1.21 GB,
  47 MXFP4 tensors.

### MLX runtime load (live probe)

`load_draft_bundle` on the QAT artifact reports `prequantized: True` with
config `{bits: 4, group_size: 32, mode: mxfp4}` and produces 47
`QuantizedLinear` modules with packed uint32 weights and E8M0 scales. MLX 0.32.4
does support the `mxfp4` mode (`_defaults_for_mode` in
`mlx/nn/layers/quantized.py` maps `mxfp4` to group size 32, 4 bits), so the
quantized draft is genuinely quantized at runtime rather than silently
dequantized.

## Why the full-logit objective could not be combined with QAT here

All B550 captures are top-k only (`full_logits: null` on `capture`,
`combined-incumbent-capture`, and `fable128-capture`). The full-vocab CE+TV
objective needs `target_logits`, which only the 896-prompt Mac capture
provides (62,026 cycles). That capture lives on the Mac, and the Mac GPU is the
only place the live gate can run, so a QAT full-logit arm requires either a new
full-logit capture on B550 or a Mac-side QAT run. The Mac cannot host CUDA QAT;
the objective and the QAT lane are therefore separable on the current
hardware, which is why the executed QAT arms use the selector objective.

## Live gates (completed)

Both gates ran on the same 128-prompt held-out corpus and target.

### RTN MXFP4 control

The previously ungated `Promoted-RTN-MXFP4` artifact scored mean acceptance
`0.674705131392368` on 128 prompts.

### QAT MXFP4

`dflash-qat-mxfp4-lr2e-6` passed exact 256-token parity
(`token_match: true`; parity smoke acceptance `0.703125`) and scored mean
acceptance `0.669362144232676` on 128 prompts.

Paired results:

| Comparison | mean delta | W/T/L | two-sided sign p |
|---|---:|---|---:|
| QAT vs BF16 incumbent | -0.010956 | 27/18/83 | 8.31e-8 |
| QAT vs RTN MXFP4 control | -0.005343 | 33/27/68 | 6.41e-4 |

Decision: rejected. QAT recovered part of the offline quantization gap but did
not transfer that recovery to live acceptance; it is worse than both the BF16
incumbent and the simpler RTN MXFP4 control.

Paired receipts:

- `build/reports/dflash/gate-qat-mxfp4-lr2e-6-heldout128-paired.json`
- `build/reports/dflash/gate-rtn-mxfp4-control-heldout128/results.json`

## Non-claims

- No QAT candidate is promoted; the live gate rejected it.
- Offline prefix/top1 are captured-cycle proxies, not acceptance, and the QAT
  offline recovery did not transfer.
- The BF16 control at LR 2e-6 not moving is a single-arm observation at one
  learning rate, not proof that BF16 cannot improve.
- The QAT arms trained on a 11,540-cycle top-k capture, not the full 62,026-cycle
  full-logit corpus.
- The `RTN MXFP4` export path is not Hessian-aware GPTQ; the QAT export reuses
  the grid the training path used, which is verified above.
