# Blackwell MXFP4 training: what the hardware buys, and what it does not

Date: 2026-10-20
Agent: Codex

## Question

Should DFlash2 QAT move to an RTX 9700 Pro Blackwell for MXFP4 tensor-core
acceleration, using the existing Blackwell container?

## Container finding

The Blackwell container in the training environment tree
(`build/reports/strix-fp4-research-20261017/flashinfer-src/docker/Dockerfile.flashinfer-ep-pytorch`)
is a **FlashInfer MoE expert-parallel (NCCL-EP) image** for multi-node
dispatch. Its own header documents that it exists to fix a cross-node
`nccl_ep` illegal-memory-access defect by rebasing onto the NVIDIA PyTorch
26.05 (CUDA 13.2) image. It is built for NCCL-EP, not for single-GPU
draft-head distillation, so it is not the container this work needs.

## Hardware finding that matters more than the container

MXFP4 tensor cores are real on SM120 and were already proven here on the Strix
RTX 5070 (SM120, CC 12.0):

- CUTLASS 4.8 `sm120_bs_gemm_mxf4_mxf4_f32_f32.cu` passed 5/5 with the
  `*mxf4t_mxf4n*` filter.
- Triton 3.8 PR #8494 supplies native SM120 `tl.dot_scaled`.
- PTX ISA 9.4 requires `sm_120a` for `.kind::mxf4` at `m16n8k64`.
- Clean NVFP4 reached 317.6 TFLOP/s at 6143 cubed on that device; DFlash2-shaped
  MXFP4 GEMMs reached 294-308 TFLOP/s at M=8192 and only 117-204 at M=128.

PyTorch's `scaled_mm` refuses MXFP4 with "only supported in CUDA for
B200/B300", but commit `95d1df7` shows that is a deliberate framework gate, not
a hardware limit.

**However, the current QAT path cannot use those cores.** The trainer's
`fake_quantize_mxfp4` quantizes the weight to the MXFP grid, then performs a
**BF16 dense matmul**. Every GEMM in training and inference runs at BF16
speed; MXFP4 is only the storage grid. On the current Ampere B550 (3090 Ti,
CC 8.6), `torch._scaled_mm` rejects MXFP4 outright and Triton `dot_scaled`
decomposes to BF16 MMA (measured 1.449 vs 77.24 TFLOP/s at the DFlash2 shape),
which is why B550 is effectively a BF16 machine for this workload.

## Decision

A 9700 Pro is **not** justified yet, for a specific reason: it would accelerate
nothing in the current code path. The quantized forward is BF16, so a faster
FP4 unit is idle. The RTX 3090 Ti already trains a full 18,666-cycle QAT arm in
about 111 seconds at 8 GB peak, so there is no throughput problem to solve.

The Blackwell path becomes valuable only if we move the QAT forward onto a
true MXFP4 matmul. That is a real piece of work with a known reference point:
on SM120, either CUTLASS 4.8 MXFP4 GEMM or Triton 3.8 `tl.dot_scaled`, with
the MXFP4 weight bytes we already produce and the E8M0 block scales we already
compute. The stale container does not provide it; the kernel comes from
CUTLASS or Triton, and PyTorch's `scaled_mm` will not do MXFP4.

## Correct order of work

The user asked to run the full-vocabulary objective first because it is
safer. That ordering is right, and the hardware decision follows from it, not
the reverse:

1. Run the full-vocab CE+TV objective through the corrected replay on the
   existing B550. The forward is BF16 there, so it is correct and cheap, and it
   answers whether the objective is sound before any kernel work.
2. Only if that objective shows live-gate promise, implement an MXFP4 matmul
   for the QAT forward (CUTLASS 4.8 or Triton `dot_scaled` on SM120) and
   measure it against the BF16 forward on the exact DFlash2 shapes.
3. Only if the MXFP4 matmul is both correct (matches the BF16 fake-quant
   forward within tolerance on real cycles) and materially faster, rent a
   Blackwell device for the QAT campaign. Verify the MXFP4 fast path with the
   same fidelity gate used for the replay, so a kernel that changes numerics
   cannot pass silently.

## Non-claims

- No Blackwell device was rented, provisioned, or benchmarked for this receipt.
- The claim that current QAT is BF16-bound follows from reading
  `mlx_gptq/qat.py::fake_quantize_mxfp4`; it was not re-profiled on this date.
- The SM120 MXFP4 capability numbers are from the 2026-10-17 Strix receipt and
  were not re-measured here.
- A future MXFP4 matmul must still reproduce the runtime draft semantics; the
  replay fidelity gate applies to the kernel change as well as the cache change.
