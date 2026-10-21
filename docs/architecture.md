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

## Contracts to preserve

- E2M1 uint32 words use MLX's LSB-first `4*i` layout; byte operands use low
  nibble first.
- Scale layout changes must be byte-compared, not inferred from throughput.
- GPU claims require live device receipts and explicit source provenance.
- The root, runtime, campaign, and CUDA checks are one quality surface.
