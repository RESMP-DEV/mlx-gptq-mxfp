# Rejected MXFP4 kernel variants, archived 2026-10-09

These are the rejected gather and FP4 kernel experiments, archived as patches
after their worktrees were retired. None of this code is canonical and none of
it should be revived without new evidence.

| Worktree | Branch | Rejection receipt | Result |
| --- | --- | --- | --- |
| `fp4-direct-float` | `research/fp4-direct-float` | `build/reports/local-ple-storage/fp4-direct-float-rejected-20261008.json` | Exact after repairing a subnormal, but the first real pair regressed p50 from 24.427 to 35.864 ms. |
| `gather-qmv-narrow` | `research/gather-qmv-narrow` | `build/reports/local-ple-storage/gather-qmv-narrow-tile-pilot-rejected-20261008.json` | Exact r=2 fast gather; only -0.014 ms short and 0.066 ms at 4,096 tokens, below the noise gate. |
| `gather-qmv-tile` | `research/gather-qmv-tile` | `build/reports/local-ple-storage/gather-qmv-wide-tile-interleaved-20261008.json` | Exact wide tile; regressed short p50 by 0.239 ms and 4,096 p50 by 0.519 ms. |

The `gather-qmm-rhs-bm` worktree from the same campaign was already restored
byte-for-byte and removed on 2026-10-09; its four tile variants are consolidated
in `gather-qmm-rhs-tiles-consolidated-rejected-20261009.json`.

The canonical geometry stays BM=32, BN=64, BK=64, WM=2, WN=2 with the r=4 fast
gather. Upstream `grouped-mm-metal` was reviewed as design input and cannot
directly substitute, because its float NAX tile needs a threadgroup tile the
quantized loader cannot hold.

## Dense MXFP4 QMV read variants, archived 2026-10-09

`dense-qmv-read-variants.patch` contains the rejected one-pack, four-pack,
shape-gated, and diagnostic-stride variants. The global four-pack build changed
real output; all isolated variants either remained below the model gate or
regressed the dominant GDN projection. The accepted alternative was the fused
GDN convolution/state kernel in mlx-lm commit `18bb2e7`, not a dense-QMV read
geometry change.
