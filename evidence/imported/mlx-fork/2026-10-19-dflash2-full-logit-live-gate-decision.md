# DFlash2 full-logit distillation: live acceptance decision

Date: 2026-10-19
Agent: Codex (main machine; training on one Prime Intellect RTX 6000 Ada pod, now terminated)

## Decision

Reject the full-logit objective candidates. No promotion. The published/promoted
incumbent `z-lab-Qwen3.8-27B-DFlash2-G3-g20.0625-g30.1875`
(`c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`) remains the
champion. Multi-GPU is not justified.

## Promotion bar

Exact 256-token target-controlled parity (`token_match true`), then paired
`heldout128` live acceptance versus the incumbent under one matched protocol,
requiring positive mean delta, wins > losses, and a two-sided paired sign test at
`p <= 0.05`. Offline CE/top1/prefix are never acceptance.

## Measured gates

Full `heldout128` gates (paired against the same incumbent control run):

| Candidate | parity | cand mean | inc mean | mean delta | W/T/L | sign p | verdict |
|---|---|---:|---:|---:|---|---:|---|
| arm1 unblended (hybrid, LR 2e-6, 17.5k cycles) | true | 0.668325 | 0.680318 | -0.011993 | 27/10/91 | 2.7e-9 | rejected |
| arm1-s025 (blend 0.25 toward incumbent) | true | 0.681700 | 0.680318 | +0.001382 | 45/45/38 | 0.51 | rejected |

32-prompt screens (subset screen, not acceptance) before committing full gates:

| Candidate | parity | mean delta (first 32) | W/T/L |
|---|---|---:|---|
| arm1-s025 | true | +0.002537 | 9/10/13 |
| arm1-s050 | true | -0.004621 | 10/7/15 |
| arm2 unblended (hybrid, LR 4e-6) | true | -0.062418 | 4/0/28 |

arm1-s025 is the only variant with a positive full-gate mean delta, but its
sign test (p=0.51) is indistinguishable from noise: wins barely exceed losses
(45 vs 38) with 45 ties. It does not meet the promotion bar.

## Why multi-GPU is not justified

Measured one-card throughput on the real full-logit hybrid objective was ~0.093
s/step at batch size 1, GPU util 87-93%, peak reserved 9.77 GB of 48 GB, with one
CPU core pinned at 100%. The run is per-cycle launch-latency bound, not memory or
FLOP bound, so a second card cannot accelerate a single sequential run; it only
helps via independent data-parallel processes. The projected full-corpus cost is
~1.5-1.8 h per arm on one Ada. Because no candidate cleared live acceptance, and
because throughput is not the constraint, scaling to multiple cards would add
cost without addressing the actual limitation (objective quality, not speed). No
2x Ada was available at decision time; this did not change the decision.

## Offline-vs-live proxy divergence (confirms prior lesson)

Higher LR kept lowering full-vocab CE (arm2 4e-6 reached CE 4.838 vs arm1 2e-6
5.736) but produced worse selector-path proxies and the worst live acceptance
(-0.062). This reproduces the prior selector campaign's finding: full-vocab CE
improvement does not transfer to live acceptance for a DFlash selector draft.
Only paired live acceptance is decisive.

## Capture used

The 896-prompt full-logit capture completed and was verified: 896 prompts,
62,026 cycles, cycle-dir count == index == receipt, `full_logits true`,
elapsed 16,457 s. The decisive arms trained on a frozen 17,500-cycle prefix
subset (256 prompt groups) that was transferred to the pod; `heldout128` is
disjoint from it by the seed-73 split.

## Non-claims

- No candidate is promoted; no rejected artifact is uploaded to Hugging Face.
- The arms trained on the 17,500-cycle subset, not the full 62,026-cycle corpus.
  A full-corpus arm is a distinct, larger run.
- Throughput/memory are one-card measurements; multi-GPU aggregate throughput was
  reasoned from the launch-bound profile, not measured on a multi-card pod.
- arm1-s025's positive mean delta is within noise (p=0.51) and is not a win.
