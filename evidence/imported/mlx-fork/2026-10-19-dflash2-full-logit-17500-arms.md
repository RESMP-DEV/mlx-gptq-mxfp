# DFlash2 full-logit distillation: 17,500-cycle decisive arms and multi-card finding

Date: 2026-10-19
Agent: Codex (main machine + one Prime Intellect RTX 6000 Ada pod)

## Scope

Train the promoted G3 DFlash2 head with the full-vocab teacher objective
(DeepSpec-style weighted CE + TV, plus the selector-path term) on a frozen
17,500-cycle slice of the live 896-prompt capture, measure one-card
throughput and memory, and decide whether multi-GPU is justified. This is a
training and scaling-decision receipt. Live acceptance is still pending and is
the only promotion authority.

## Objective and trainer state

- Loss mode `hybrid`: selector CE (top-16 teacher path) plus full-vocab teacher
  CE and TV against the reconstructed Qwen3.8-27B BF16 LM head, weights
  `selector 1.0`, `CE 0.1`, `TV 0.9` (DeepSpec balance).
- Distribution/hybrid modes replay the full draft transformer so LoRA receives
  gradients; selector-only mode uses the captured hidden shortcut. Fixed after
  the first hybrid run silently trained only the selector projection.
- Trainer hardening committed to `RESMP-DEV/mlx-gptq-mxfp` branch
  `work/dflash2-calibration`: metric-totals init (`938cb99`), full-replay path
  (`5a3d204`), CUDA memory telemetry (`12254e5`), explicit loss weights
  (`ffba935`), finalization/QAT-export/fingerprint hardening (`5f47153`),
  candidate merge tool (`970c63b`). Local suite: 46 passed, 5 skipped.

## Data

- Frozen subset `fable-full-logits-frozen17500`: cycles 0..17499 of the active
  capture, 256 prompt groups, materialized on the pod by symlink to the
  transferred immutable cycle dirs (no data copy).
- Source capture `fable-coding-896-full-logits-capture` (896 train prompts,
  disjoint `heldout128` live gate) was still growing during training; the frozen
  index is a fixed prefix, so training never saw a moving corpus.
- No `heldout128` prompt appears in the training subset (disjoint by the
  seed-73 split).

## LR gate (frozen 4,096-cycle snapshot, hybrid, seed 73)

| LR | best val CE | best prefix | best top1 |
|---|---:|---:|---:|
| 5e-7 | 7.152 | 0.727 | 0.811 |
| 1e-6 | 7.137 | 0.739 | 0.820 |
| 2e-6 | 7.069 | 0.754 | 0.833 |
| 4e-6 | 6.615 | 0.742 | 0.852 |
| 8e-6 | 5.678 | 0.742 | 0.852 |

Only `2e-6` clears the 1% validation-CE gate among the low three. Higher LR
keeps lowering full-vocab CE but the selector-path prefix proxy peaks near
`2e-6` and does not improve beyond it.

## Decisive 17,500-cycle arms (hybrid, seed 73, 1 epoch, one Ada)

| Arm | LR | best CE | best top1 | best prefix | peak GPU | wall |
|---|---:|---:|---:|---:|---:|---:|
| hybrid-17500-2e-6-e1 | 2e-6 | 5.736 | 0.885 | 0.793 | 9.77 GB | 1252.9 s |
| hybrid-17500-4e-6-e1 | 4e-6 | 4.838 | 0.875 | 0.777 | 9.77 GB | ~1250 s |

Both start from the same incumbent (init CE 7.595, prefix 0.764, top1 0.848).
Arm2 (4e-6) reaches a lower full-vocab CE but worse selector-path proxies; arm1
(2e-6) has the better prefix/top1. This is the same proxy-divergence pattern
that made offline loss a poor acceptance predictor in the prior selector
campaign, so neither offline number is treated as acceptance.

## Merged candidates (local, CPU, byte-reproducible merge discipline)

- Base: `z-lab-Qwen3.8-27B-DFlash2-G3-g20.0625-g30.1875`
  (`c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`).
- arm1 merged: `da51e3f6e01347440dc87bd4dcc110b3b6dc8f564f6b1bb3ae1d09b72e850882`.
- arm2 merged: `98bf1b7db46489148f891f2999aa6325d1cef1a01a7c9f2793b9b4de82eeffb4`.
- Adapters pulled back hash-verified (arm2 `ba404aaf...` matches pod).

## One-card scaling finding and multi-GPU decision

- Measured single-card throughput on the real full-logit hybrid objective:
  ~0.093 s/step at batch size 1, 16,587 steps in ~1253 s. GPU util 87-93%,
  peak reserved 9.77 GB of 48 GB, one CPU core pinned at 100%.
- The run is per-cycle launch-latency bound, not GPU-memory or FLOP bound. A
  second card cannot accelerate one sequential batch-1 run; it only helps via
  independent data-parallel processes (~2x aggregate) or a large parallel
  experiment matrix.
- Projected full-corpus cost on one Ada: the ~70-80k-cycle 896-prompt capture
  is ~1.5-1.8 h per arm. Multi-GPU is therefore not justified by throughput for
  a single decisive arm.
- Decision: stay on one card for the decisive arm. Reconsider multi-GPU only
  after a candidate wins live acceptance AND the next step is a broad parallel
  LR/rank/weight matrix or a materially longer corpus, where process-level
  parallelism would shorten wall clock. No 2x Ada was available at decision
  time (2x A6000 and H100 pairs were); this did not block the decision because
  throughput is not the constraint.

## Pending (not claimed here)

- Live acceptance: exact 256-token parity plus paired `heldout128` mean
  acceptance versus the incumbent, on the Mac GPU after the capture frees it.
  Gate script `build/reports/dflash/live_gate_full_logit.sh`.
- Promotion requires parity `token_match true` and positive paired live delta.
  Offline CE/top1/prefix are not acceptance.

## Non-claims

- No candidate is promoted; live acceptance has not been run.
- Offline proxy improvement does not prove live acceptance (prior campaign
  showed proxies can improve while acceptance does not).
- The 17,500-cycle subset is a fixed prefix of a still-growing capture, not the
  full 896-prompt corpus; a full-corpus arm is a separate, larger run.
- Throughput/memory are one-card measurements; multi-GPU aggregate throughput
  was reasoned from the launch-bound profile, not measured on a 2-card pod.

## Update: completed capture and live rejection of arm1

The 896-prompt capture completed at 20:52 UTC. Verification found:

- prompt count: 896;
- indexed cycles: 62,026;
- cycle directories: 62,026;
- `full_logits: true`;
- elapsed capture time: 16,457.389 seconds.

The watcher then ran the real Mac live gate for arm1:

- exact 256-token parity: `token_match: true`;
- parity smoke acceptance: `0.703125`;
- candidate `heldout128` mean acceptance: `0.6683250000501196`;
- incumbent mean acceptance: `0.6803182327226998`;
- mean delta: `-0.011993232672580012`;
- profile: 27 candidate wins, 10 ties, 91 losses;
- two-sided sign-test p-value: `2.6899026850690616e-9`.

Decision: rejected. Exact parity alone is not acceptance, and the large offline
CE/top1/prefix gains did not transfer to live acceptance. Paired receipt:
`build/reports/dflash/gate-arm1-2e-6-heldout128-paired.json`.

This rejection also removes the current justification for multi-GPU spend: the
full-logit candidate is not yet live-superior even on one card.
