# DFlash2 Blackwell container and operand-precision boundary

Date: 2026-10-20
Agent: Codex

## Question

Can the Blackwell container in `RESMP-DEV/tune-env` make the current DFlash2
QAT path use MXFP4 tensor cores on an RTX 9700 Pro-class SM120 device?

## Container state

The relevant image is `tune-env:blackwell-sm120-cuda13.0.1-ubuntu24.04`, built
from `tune-env/docker/Dockerfile.flash-blackwell-sm120`. It contains PyTorch
2.14.0+cu130, Triton 3.8.0, and FlashAttention 2.8.4 built from official FA4
beta35 commit `1fc02211`. Its 2026-10-19 receipt records a live causal BF16
FlashAttention forward pass on Strix RTX 5070 Laptop SM120.

The image is useful as a kernel-development base, but it is not a DFlash2 QAT
environment. It has no DFlash2 MXFP4 forward/backward wrapper, no CUTLASS tree,
and its FlashAttention smoke is not FP4 evidence. The separate Jina
`matmul_ogs` result is also not DFlash2 evidence; that kernel was slower than
BF16 on the measured Jina shapes.

Live re-probe on 2026-10-20 was blocked: Strix SSH timed out and B550 SSH
closed during banner exchange. No current GPU visibility or container claim is
made from this date.

## Operand-precision boundary

The pinned Triton 3.8 block-scaled tutorial was retrieved from
`https://raw.githubusercontent.com/triton-lang/triton/v3.8.0/python/tutorials/10-block-scaled-matmul.py`.
Its kernel has three paths:

- FP8 E4M3 LHS with MXFP4 RHS for mixed precision;
- MXFP4 LHS with MXFP4 RHS;
- FP8 E4M3 with FP8 E4M3.

There is no BF16 LHS with MXFP4 RHS path. Consequently, the current DFlash2
QAT implementation cannot use native FP4 tensor cores merely by replacing its
matmul: its activations are BF16 and its fake-quantized MXFP4 weights are
dequantized back into a BF16 dense matmul.

This also explains why the prior matched SM120 FP4×FP4 benchmark is not yet a
QAT speed result. It proved native tensor-core execution for block-scaled FP4
operands, not the deployed weight-only W4/BF16 semantics.

## Required order

1. Finish the corrected full-vocabulary objective and confirm nonzero LoRA
   updates and stable validation behavior.
2. Run matched BF16/RTN and QAT controls on the same session-disjoint capture.
3. If QAT is promising, implement a block-scaled forward with an explicit
   activation operand policy: MXFP8×MXFP4 first, then optionally MXFP4×MXFP4.
4. Prove forward numerics against the BF16 fake-quant reference on real cycles,
   not random cubes.
5. Implement and prove backward gradients. The Triton tutorial is forward-only;
   importability or forward speed does not qualify QAT.
6. Benchmark exact DFlash2 shapes and include packing, activation quantization,
   and backward costs. Only a positive result justifies 9700 Pro/SM120 time.

## Non-claims

- No Blackwell GPU was rented or purchased and no container was executed on
  2026-10-20.
- The 2026-10-19 FA4 and Jina receipts remain historical, live evidence for
  their own workloads only.
- No claim is made that MXFP8 activations preserve DFlash2 quality; that is an
  experiment gate, not an assumption.
- No model artifact is promoted or uploaded by this research update.
