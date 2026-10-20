# SM120 native MXFP4 DFlash2 kernel qualification

Date: 2026-10-20
Agent: Codex
Host: Prime pod `7b2211a4449a46fdaaa7a7d0b2ca1d6c` (terminated after run)

## Live host and stack

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Compute capability: 12.0
- VRAM: 97,887 MiB
- Driver: 580.126.09
- PyTorch: 2.14.0+cu130
- Triton: 3.8.0
- CUDA reported by Torch: 13.0

The pod was on-demand, not spot. It was terminated after receipts and PTX were
copied back. The bounded run reduced the Prime wallet from `$37.38` to `$36.75`.

## Kernel path

`tools/mxfp4_kernel/mxfp4_matmul.py` implements a Triton
`tl.dot_scaled` MXFP4 by MXFP4 matmul on exact DFlash2 projections. It:

- consumes the DFlash2 packer's packed uint32 E2M1 words and uint8 E8M0 scales;
- repacks E2M1 two codes per byte for Triton;
- converts row-major scales to Triton's canonical 128-row scale bands;
- pads M to 128 for decode row counts;
- keeps A and B scale bands separate;
- verifies against a dequantized BF16 reference;
- inspects compiled PTX and fails unless `kind::mxf4nvf4` is present.

A live mixed MXFP8 by MXFP4 probe compiled to 256 BF16 MMAs and no native MXFP
instruction. That path was rejected. The qualified path quantizes both operands
to MXFP4.

## Correctness and native instruction proof

The 128x128x512 probe was exact (`max_abs_error = 0`,
`max_rel_error = 0`). A multi-output-tile bisect then exposed a B-scale band
using `pid_m`; after changing it to `pid_n`, the following passed:

```text
N=128 K=512   rel=0
N=256 K=512   rel=0
N=512 K=512   rel=0
N=512 K=1024  rel=0
N=512 K=2048  rel=0.006369
N=4096 K=512  rel=0
```

All 16 dominant exact DFlash2 shape cases passed the harness gate. The final PTX
contains 64 `mma.sync...kind::mxf4nvf4...e2m1.e2m1...ue8m0` instructions and
zero BF16 MMAs.

Artifacts:

- `build/reports/dflash/sm120-mxfp4-20261020/exact-results.json`
- `build/reports/dflash/sm120-mxfp4-20261020/bf16-control-results.json`
- `build/reports/dflash/sm120-mxfp4-20261020/speedup.json`
- `build/reports/dflash/sm120-mxfp4-20261020/final-kernel.ptx`

## Kernel-only results

| Shape family | M=5 speedup | M=128 speedup | M=2048 speedup | M=8192 speedup |
| --- | ---: | ---: | ---: | ---: |
| fc | 1.35x | 1.65x | 3.26x | 3.50x |
| mlp.down_proj | 1.33x | 1.68x | 3.21x | 3.41x |
| mlp.gate_proj | 3.29x | 3.70x | 3.14x | 3.43x |
| self_attn.q_proj | 0.57x | 0.87x | 2.84x | 3.42x |

Best logical kernel throughput was 1435.28 TFLOP/s on fc at M=8192. The matched
BF16 control reached 409.75 TFLOP/s on that shape. Isolated launch overhead can
make very small Q-projection cases slower than BF16.

One-time preparation (activation MXFP4 quantization plus weight conversion) is
reported separately in `prepare_ms`; it is not included in kernel-only timing.
For example, fc M=8192 prepare was 52.13 ms in this first Python implementation.
That preparation must be fused or amortized before any end-to-end training
claim.

## Non-claims

- This is an isolated synthetic GEMM qualification, not DFlash2 model quality.
- Activation quantization to E2M1 may change model outputs; no real-cycle
  fidelity or live acceptance was run.
- No backward path exists yet. This is not QAT integration.
- Kernel-only speedups do not predict end-to-end training or serving speed.
- No model artifact is packed, promoted, or uploaded.
