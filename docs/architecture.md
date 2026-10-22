# MLX Stack architecture

## Philosophy

Correctness precedes speed: bit-exact layouts and explicit numerical gates come
first, then bounded-memory preparation, then measured kernel throughput, and
finally maintainability of the consolidated stack. A performance result without
provenance and non-claims is not accepted.

## Ownership

| Boundary | Owner | Current position |
|---|---|---|
| Calibration engine | Root `mlx_gptq/` and `tools/` | Active |
| DFlash MLX runtime | `runtime/dflash-mlx/` | Subtree-merged lineage |
| Qwen campaigns | `campaigns/mlx-gptq-calibration/` | Subtree-merged lineage |
| CUDA MXFP probes | `cuda/mlx-cuda-mxfp/` | Subtree-merged lineage |
| Upstream MLX fork | `RESMP-DEV/mlx` | Pure fork; not dumped into here |
| MLX runtime contract | `RESMP-DEV/mlx-lm` | External contract |

## Open choices

| Topic | Current position | Decision gate |
|---|---|---|
| Default branch promotion | `work/dflash2-calibration` is complete; `main` is older | `just quality` green after subtree merges |
| Old repository retirement | Historical repositories retained read-only as subtree sources; redundant local clones removed | New work targets `mlx-stack` only |
| Full projection matrix | Dominant activation/GEMM cases qualified; all weights qualified | Extend only when a missing shape owns measured time |
| Local 27B CCR fallback | Calibrated MXFP4 loopback server qualified at 32 short-context requests | Promote only through source-policy review and idle listener restart |

## 2026-10-20 work log

- Renamed the owned engine repository from `mlx-gptq-mxfp` to `mlx-stack`,
  preserving GitHub redirects and updating local remotes.
- Subtree-merged `dflash-mlx`, `mlx-gptq-calibration`, and `mlx-cuda-mxfp`.
- Added fused SM120 activation MXFP4 packing, direct uint32 E2M1 weight
  repacking, layout regression coverage, and the Tetra evidence receipt
  `evidence/2026-10-20-tetra-fused-mxfp4-activation-weight-packing.md`.
- Added the stack quality dispatcher and component ownership map.
- The full quality dispatcher passed after an M4-specific DFlash verify-kernel
  deviation ceiling was pinned; receipt summary:
  `evidence/2026-10-20-mlx-stack-consolidation.md`.
- Imported the campaign evidence formerly tracked in `RESMP-DEV/mlx`, merged
  the pushed Qwen parity-tooling branch, and removed the three verified-clean
  redundant local clones. `/Users/kearm/mlx-stack` is the convenience entry to
  the canonical checkout.
- Measured the local calibrated Qwen3.8-27B MXFP4 artifact through an isolated
  CCR OpenAI route: 142 aggregate tok/s at 32 short-context requests, but only
  about two useful concurrent 8k prefills. Receipt:
  `evidence/2026-10-20-local-27b-ccr-fallback-capacity.md`.
- Consolidated four bounded EAGLE-4 audits into the correctness-first Qwen
  speculative-decoding plan. The pinned draft, official inference source,
  paper, target-vocabulary contract, low-rank/full-head behavior, and Qwen GDN
  verification constraint are recorded in
  `evidence/2026-10-20-eagle4-qwen38-27b-mlx-port-research.md`.
- Added the first EAGLE-4 MLX draft model and strict loader/schema gate. The
  real Qwen checkpoint lazy-loads all 19 BF16 tensors and passes a one-token
  Metal full-head/candidate-head smoke; the existing DFlash engine backend is
  explicitly gated until the EAGLE adapter is built. See the implementation
  section of the same evidence receipt.

## 2026-10-21 work log

- Added the correctness-first EAGLE-4 chain backend to the DFlash engine,
  including absolute target positions, exact candidate-ID rescoring, greedy
  suppression, and explicit rejection of unsupported tree/sample paths.
- Parent review caught and repaired the initial zero-based RoPE positions;
  official DraftExtend uses `seq_len + j`. An independent review approved the
  pairing and mapping, and its sparse-context and greedy-path repairs are
  included.
- Measured corrected EAGLE-4 b2 at 8.618 tok/s on BF16 and 33.568 tok/s on
  the owned calibrated MXFP4 target, with exact native greedy IDs. The
  incumbent MXFP4 DFlash2 stack remains best at 51.028 tok/s. Full details,
  commands, and non-claims:
  `evidence/2026-10-21-eagle4-bf16-mxfp4-serving-comparison.md`.
- Added the first Flash-Next native-MTP target substrate without altering the
  27B/DFlash2 trainer. `NativeMTPBank` validates Strata's aligned full-logit,
  10,240-wide residual, and sidecar files; `NativeMTPPairDataset` maps 512
  accepted verifier rows to 511 two-ahead MTP pairs by pairing residual row
  `i` and token `i+1` with teacher logits row `i+1`. The real private bank and
  all 511 pairs passed on Strix. See
  `evidence/2026-10-21-flashnext-mtp-bank-loader.json`.
- Implemented and exercised the native MTP itself. Transformers 5.18 ignores
  `mtp.*`, so `FlashNextMTP` loads all 31 MTP tensors plus the shared embedding
  and head and reuses the trunk's exact attention, gated-residual, and MoE
  blocks. On captured row 0/token 795 it emits argmax 6195, exactly the next
  sidecar input token, validating the two-ahead chain. A compact 33-tensor
  training checkpoint is 7,757,104,472 bytes.
- Added `FlashNextMTP` MXFP4 QAT with float32 LoRA masters over 18 selected
  dense/shared modules, absolute source-position RoPE inputs, full-vocabulary
  CE plus TV, dynamic FP16 scaling for a future T4 run, exact effective-grid
  export, and offline-to-Forge W&B sync. A
  real 16-step CPU QAT slice improved same-prompt validation CE from 11.045 to
  9.036 and top-1 match from 0.98% to 5.88%. This is not live acceptance.
- The T4 lane was allocation-first and Colab assigned T4s three times, but
  `exec`, `console`, SSH, and fresh `run` all failed in the same Jupyter
  WebSocket path while HTTP upload worked. No T4 training step is claimed. For
  this MTP-only path, eager/SDPA is the sound backend; compiled generic
  FlexAttention can execute on T4 but has no model-level integration or benefit
  at one-token sequence length. `wandb` is the only added dependency. Full
  evidence: `evidence/2026-10-21-flashnext-mtp-mxfp4-qat.json`.
- Added explicit non-reentrant gradient checkpointing to the custom MTP forward.
  Strix is reachable and hosts only its embedding service, but that service
  leaves about 5.6 GiB while the frozen FP16/BF16 base alone is 7.22 GiB, so
  checkpointing cannot make the full GPU-resident path fit there. A real
  16-step Strix CPU checkpointing run passed in 3:06 without disturbing GPU
  occupancy.
- Selected an on-demand Prime A6000 48 GB at $0.54/hour after confirming the
  recurring Modal credit lane was exhausted. The A6000 loaded the pinned Hub
  source and hash-verified teacher bank, completed matched 16-step arms with
  and without checkpointing, then completed the full 409-step FP16 MXFP4 QAT
  epoch. Peak sampled VRAM was 17,464 MiB; same-prompt CE improved from 11.025
  to 2.473 and top-1 to 60.78%. The pod was terminated after artifact retrieval
  for $0.73 total. Full evidence:
  `evidence/2026-10-21-flashnext-mtp-strix-a6000.json`.

- Evaluated the Flash-Next MTP MXFP4 QAT candidate against its frozen base
  on a session-disjoint teacher bank collected from a different prompt, with
  the effective weight rebuilt from the stored MXFP4 codes and scales. On the
  held-out prompt, cross-entropy fell from 7.836 to 4.374 and top-1 agreement
  rose from 7.63% to 18.40%, with 70 wins against 15 losses over 511 pairs
  and an exact McNemar p of 1.17e-09. The same-prompt gain is much larger
  (top-1 60.47%), so the difference is partly memorization and the held-out
  numbers are the honest ones. A 16-step variant of the same recipe did not
  generalize (held-out top-1 9.00%, p of 0.21). The full-epoch artifact was
  published as a private first draft to
  `RESMP-DEV/Qwen3.8-Flash-Next-MTP-MXFP4-QAT-draft`, and the published
  artifacts were downloaded back and hash-verified. This is not a serving
  candidate and no live acceptance was measured. Full evidence:
  `evidence/2026-10-21-flashnext-mtp-heldout-evaluation.json`.

- Answered whether the MTP QAT gain is memorization or specialization, and
  rebuilt the recipe accordingly. Trained on nine prompts from four corpora
  and measured on four prompts whose group identifiers are disjoint from
  every training prompt, the mixed head reaches 38.41% held-out top-1 against
  a 4.84% base, with 709 wins and 23 losses over 2,044 pairs and an exact
  McNemar p of 1.9e-177. Its same-corpus top-1 is 40.81%, so the gap between
  fitted and held-out performance fell from 42.1 points to 2.4: the gain
  transfers across prompts, which is specialization to the target's draft
  distribution rather than memorization of one prompt. Teacher confidence
  clamping was implemented and measured at floors 0.98 and 0.995; both cost
  held-out top-1, so the published artifact uses the exact verifier
  distribution. The improved candidate replaced the private first draft at
  `RESMP-DEV/Qwen3.8-Flash-Next-MTP-MXFP4-QAT-draft` and the published
  artifacts were downloaded back and hash-verified. Full evidence:
  `evidence/2026-10-22-flashnext-mtp-mixed-prompt-generalization.json`.

## 2026-10-22 Flash-Next MTP production validation

The mixed-prompt draft was independently validated from its live private Hub
revision. All four published files hash-match, its 54 LoRA and 36 MXFP4
tensors are finite and cover 18 modules, and a fresh M4 Max MPS FP16 SDPA
evaluation of all 2,044 group-disjoint pairs reproduced the effect: base top-1
4.8434%, candidate 38.3072%, 707 wins versus 23 losses, and exact McNemar
`p=7.17e-177`. The focused loader, evaluator, mixer, trainer, and export
tests passed 31/31. Cross-device differences from the A6000 receipt are small
and do not change the result.

It is not production-usable yet. Strata loads `dense.txt`, `dense.bin`, and
`experts.bin`, with no safetensors, LoRA, or MXFP4 dense-weight path and no
converter from this artifact. More importantly, training optimized independent
single-cell forwards without KV history while Strata's drafter attends over its
own KV window (default 32,768 cells); the candidate also tunes an indexer
projection the runtime dense path does not consume, and training used BF16
shared components while serving binds the deployed quantized target and may
restrict proposals through a draft vocabulary. No PyTorch/CUDA parity check,
live speculative acceptance, throughput A/B, rollout flag, or rollback receipt
exists. The verdict is therefore offline-validated but production NO-GO. Full
evidence: `evidence/2026-10-22-flashnext-mtp-production-validation.json`.

## 2026-10-22 Flash-Next MTP matching-bpw format selection

The trained MXFP4 grid stores 17 bytes per 32 weights (4.25 bpw). Q4_0,
IQ4_NL, and Q4_K all consume 4.5 bpw, while IQ4_XS consumes 136 bytes per 256
weights and is the exact 4.25-bpw GGML geometry. Because four tuned matrices
have 320/640-column rows, IQ4_XS cannot cover 12.855% of the tuned weights;
the measured policy uses IQ4_NL for those four tensors and averages
4.28214 bpw.

All comparisons used the production `ggml_quantize_chunk` entrypoint without
an importance matrix and the same 2,044 group-disjoint pairs. Base top-1 is
4.8434% and MXFP4 is 38.3072%. The mixed IQ4_XS policy reaches 37.867%,
retaining 98.68% of MXFP4's absolute top-1 gain. Q4_0 reaches 37.622%, IQ4_NL
37.476%, Q4_K 37.280%, and IQ3_S collapses to 25.783%. Q4_0 has slightly lower
cross-entropy, but IQ4_XS is the proper matching-bpw target for argmax draft
transfer because it has the best top-1 result at less effective storage.

The selected packed artifact contains all 18 tuned modules, covers
89,210,880 weights in 47,754,584 bytes, and round-trips from its saved blocks
to the same metrics. `IQ3_XS` is only a llama-quantize alias for IQ3_S in this
pinned GGML stack, not a distinct geometry. This is still not a complete
Strata rt_dir and does not measure live acceptance. Full evidence:
`evidence/2026-10-22-flashnext-mtp-iq4xs-format-selection.json`.

## Contracts to preserve

- E2M1 uint32 words use MLX's LSB-first `4*i` layout; byte operands use low
  nibble first.
- Scale layout changes must be byte-compared, not inferred from throughput.
- GPU claims require live device receipts and explicit source provenance.
- The root, runtime, campaign, and CUDA checks are one quality surface.
