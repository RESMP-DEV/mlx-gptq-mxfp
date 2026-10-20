# DFlash2 selector distillation live gate

Date: 2026-10-17

## Scope

This records target-aware distillation experiments for the campaign-owned
native-MXFP4 DFlash2 draft. Every promoted artifact must preserve exact
target-controlled token output and improve measured live acceptance. Training
loss, offline replay, or throughput alone is not acceptance.

The final verdict is rejection of every selector-only distilled projection in
this document. The valid delivered draft remains:

`/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-1981cycles`

No distilled DFlash2 artifact was uploaded to Hugging Face. Strix was not used.

## Infrastructure and code

B550 (`compute`, 192.168.1.171) was reachable with two idle RTX 3090 Ti GPUs.
The transferred BF16 draft SHA-256 matched the local source:

`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`

The complete 1,981-cycle transformer-replay corpus, DFlash PyTorch source at
`07ebd93`, and private quantizer code were copied to
`/home/kearm/dflash2-distill`. A four-cycle CUDA smoke passed. Existing sweep
jobs were stopped only after the first objective defect was proven; their status
files record the aborted edge-only runs.

Private code changes:

- `RESMP-DEV/dflash-mlx@113af61..a7027c2`: capture the selector input hidden
  state and support a compact selector-only capture mode.
- `RESMP-DEV/mlx-gptq-mxfp@e3330ce..a288897`: include captured target unary
  scores and runtime top-k in the objective, initialize from a projection file,
  replay captured selector hidden states, and add mean or sequence-sum loss.

Validation:

- DFlash runtime bundle tests: 14 passed.
- Selector objective tests on B550: 2 passed.
- Ruff, formatting, Python compilation, and `git diff --check` passed for the
  changed modules.

## Objective corrections

The first trainer optimized edge-only cross entropy over 64 candidates. This
did not match inference, which selects from 16 candidates using
`target_logit + edge`. The corrected objective uses captured target logits,
the model's runtime top-k, and the additive score. A synthetic test proves the
unary score affects the loss and that an absent teacher token is rejected.

The first domain correction replayed the BF16 transformer even though the live
artifact uses calibrated MXFP4 weights. The capture harness now records the
selector input hidden state from the real calibrated MXFP4 runtime. Training
then starts from the dequantized artifact projection rather than the unrelated
BF16 source projection.

## Captures

Selector-only capture from the calibrated MXFP4 draft:

- run 1: 1,980 cycles, 24 prompts, 256 tokens each, 233.140 s, 73 MB;
- run 2: 1,998 cycles from the first live domain-distilled artifact, 277.954 s,
  82 MB before transfer.

The compact captures contain the anchor, real selector hidden state, target
top-64 IDs and logits, and target greedy IDs. They do not contain the unused
raw target context, which accounts for the size reduction from the 1.2 GB
transformer-replay corpus.

## Live gates

The calibrated MXFP4 control had 72.656% acceptance on the standard 256-token
smoke prompt, 68.490% mean acceptance on the three-prompt held-out set, and
66.683% mean acceptance over the 24-prompt corpus.

| Candidate | Exact 256-token parity | Smoke acceptance | Held-out mean | Corpus result |
| --- | ---: | ---: | ---: | --- |
| MPS edge-only, 2 epochs | pass | 70.703% | 67.188% | 65.479%, 3 wins / 4 ties / 17 losses |
| B550 edge-only, e16 lr 2e-5 | pass | 69.922% | 66.146% | not promoted after regression |
| B550 additive, e8 lr 1e-5 | pass | 69.531% | 67.188% | not promoted after regression |
| Domain-matched, e8 lr 1e-6 | pass | 71.094% | 68.490% (tie) | 66.406%, 8 wins / 4 ties / 12 losses |
| Domain-matched, e4 lr 2e-6 | pass | 71.094% | 67.969% | not promoted after regression |
| On-policy mean, e4 lr 5e-7 | pass | 70.313% | 67.188% | not promoted after regression |
| On-policy sequence-sum, e4 lr 5e-7 | pass | 70.313% | 67.188% | not promoted after regression |

Every exact-parity gate listed above matched all emitted target IDs. This
validates safety under target verification, not draft quality.

## Why proxy metrics improved but live acceptance did not

The corrected domain runs improved held-out cycle-validation loss from
0.748841 to 0.617828. Offline accepted-prefix replay on the first capture
improved full-block agreement from 59.293% to as high as 72.626%.

The second on-policy capture improved its own validation metrics again:

- mean reduction: 0.427788 to 0.418997;
- sequence-sum reduction: 1.340201 to 1.310818;
- offline full-block agreement: 68.719% to 71.822%.

Changing the selector changes later draft blocks and therefore changes the
hidden-state distribution. Static replay and cycle-validation loss remained
off-policy for the final trajectory. Live held-out and corpus acceptance were
the deciding gates and rejected the projections.

## Receipts and artifacts

Representative receipts:

- MPS run: `build/reports/dflash/distill-full-mps/receipt.json`
- MPS artifact: `models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-distilled-selector`
- MPS corpus pairing:
  `build/reports/dflash/distilled-selector-mxfp4-corpus24/paired_vs_mxfp4.json`
- domain run:
  `build/reports/dflash/b550-domain-e8-lr1e-6/receipt.json`
- domain corpus pairing:
  `build/reports/dflash/domain-distilled-selector-e8-lr1e-6-corpus24/paired_vs_mxfp4.json`
- second capture:
  `build/reports/dflash/selector-capture-run2/receipt.json`
- sequence-sum run:
  `build/reports/dflash/b550-onpolicy-sum-e4-lr5e-7/receipt.json`

All rejected artifact directories remain local and clearly named. They are not
durable model releases.

## Non-claims

- No broad quality or user-serving claim is made from 24 prompts and three
  held-out prompts.
- No selector codebook was quantized.
- No full DFlash2 transformer distillation was attempted.
- No Optane-specific result is claimed.
- B550 was left with no active DFlash2 tmux session after the sweep.
