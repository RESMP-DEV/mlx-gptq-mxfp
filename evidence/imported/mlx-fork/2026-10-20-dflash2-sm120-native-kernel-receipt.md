# DFlash2 SM120 native MXFP4 kernel receipt

Date: 2026-10-20
Agent: Codex
Implementation: `RESMP-DEV/mlx-gptq-mxfp` branch `work/dflash2-calibration`,
head `063f72c`.

## Result

A live on-demand RTX PRO 6000 Blackwell Server Edition (SM120, CC 12.0, 96 GB)
ran a Triton 3.8 `tl.dot_scaled` MXFP4 by MXFP4 matmul on exact DFlash2
projection shapes. All 16 dominant shape cases passed numerical verification.
Compiled PTX contains 64 native
`mma.sync...kind::mxf4nvf4...e2m1.e2m1...ue8m0` instructions and zero BF16
MMAs.

A mixed MXFP8 by MXFP4 probe compiled to BF16 emulation on SM120 and was
rejected. The qualified path quantizes both operands to MXFP4.

Matched BF16 controls show 3.1x to 3.5x kernel-only speedups at M=2048 and
M=8192. Peak logical throughput was 1435.28 TFLOP/s versus BF16's 409.75
TFLOP/s on the same fc shape. Very small Q-projection cases can be slower
because a 128-row padded native tile dominates launch/work ratio.

The activation-quantization and weight-repack preparation is currently separate
and Python-side (for example, 52.13 ms for fc M=8192). It must be fused or
amortized before claiming end-to-end QAT speedup.

The task-owned Prime pod was terminated after receipts were copied. Wallet cost
for the successful run was about `$0.63`.

## Validation

- Host: RTX PRO 6000 Blackwell Server Edition, CC 12.0, driver 580.126.09.
- Stack: PyTorch 2.14.0+cu130, Triton 3.8.0.
- Repository checks: `70 passed, 5 skipped`; kernel harness ruff clean.
- Live correctness: all 16 exact dominant DFlash2 shapes passed.
- PTX: 64 `kind::mxf4nvf4`; zero BF16 MMA.
- Ignored receipts:
  `/Users/kearm/mlx/build/reports/dflash/sm120-mxfp4-20261020/`.

## Non-claims

- Synthetic operands prove kernel numerics and throughput only, not DFlash2
  model quality.
- MXFP4 activations have not been checked against real captured cycles.
- No backward implementation, real-cycle fidelity gate, live acceptance, or
  model promotion has been run.
