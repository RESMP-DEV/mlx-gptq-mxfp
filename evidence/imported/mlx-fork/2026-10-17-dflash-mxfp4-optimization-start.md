# DFlash2 MXFP4 optimization start

Date: 2026-10-17

## Scope

This records the first optimization pass for
`RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4` with
`z-lab/Qwen3.8-27B-DFlash2`. The target checkpoint and its target-verification
semantics are fixed. The measurements cover one M4 Max and are not a
multi-user serving benchmark.

## Runtime and accepted integration fix

The private runtime is `RESMP-DEV/dflash-mlx`, branch
`work/local-mxfp4-dflash2`, commit
`9ced85b5e01a06ea5cbeda8ca026ec43735bc1d5`.

That commit repairs local target/draft references so a local calibrated MXFP4
target plus local DFlash2 checkout applies the model-default affine W4 draft
quantization. Without an explicit flag, the real bundle now reports:

- `effective_draft_quant=w4`;
- `draft_quant_spec=w4`;
- `draft_quant_source=model_default`.

Validation was 13 focused runtime-bundle tests plus a real 128-token smoke.

## Phase attribution

The three-repeat 256-token W4/group-64 run had median baseline 26.385 tok/s
and DFlash2 53.156 tok/s. Its middle-run phase totals were:

| Phase | Milliseconds |
| --- | ---: |
| Target prefill | 893.060 |
| Draft total | 225.103 |
| Draft prefill | 15.636 |
| Draft incremental | 209.467 |
| Target verify | 161.162 |
| Replay | 11.048 |
| Commit | 26.296 |

The adaptive policy spent 53 cycles at block four and 18 cycles at block five.
Block five had higher tokens-per-cycle and tokens-per-second in this run.

## Block sweep

Three-repeat, 256-token DFlash-only measurements on the default math prompt:

| Block | Median tok/s | Acceptance | Peak |
| --- | ---: | ---: | ---: |
| 4 | 54.232 | 71.484% | 19.672 GB |
| 5 | 53.009 | 72.266% | 19.672 GB |
| 6 | 52.510 | 72.266% | 19.672 GB |
| 8 | 50.033 | 72.266% | 19.689 GB |

Blocks six and eight are capped by DFlash2's maximum block size. Exact parity
checks at blocks four, five, and 16 all matched 256/256 target token IDs.

A separate three-prompt custom suite contradicted the single math prompt:

| Block | Median DFlash tok/s |
| --- | ---: |
| 5 | 44.010 |
| 4 | 35.867 |

Therefore no global default-block change was accepted. The adaptive policy
remains the correct default across prompt types.

## Draft quantization sweep

Three-repeat, 256-token DFlash-only measurements:

| Draft format | Median tok/s | Acceptance | Peak |
| --- | ---: | ---: | ---: |
| W4/group-64 | 53.156 | 72.266% | 19.688 GB |
| W4/group-32 | 51.907 | 71.875% | 19.785 GB |
| W4/group-128 | 52.595 | 72.266% | 19.617 GB |
| W8/group-64 | 51.208 | 72.656% | 20.588 GB |
| W8/group-128 | 47.849 | 72.266% | 20.533 GB |

W4/group-64 remains the default. None of these alternatives is accepted.

## BF16 LM-head measurement

The calibrated target retains an untied BF16 language-model head of shape
`248320 x 5120`. Isolated calls measured:

| Input rows | Milliseconds per call |
| ---: | ---: |
| 1 | 5.611 |
| 4 | 5.519 |
| 5 | 5.586 |
| 8 | 7.710 |

The sparse-head implementation map is recorded by the read-only worker, but no
sparse candidate path was implemented in this pass. A candidate set smaller
than the full vocabulary can change acceptance and therefore emitted target
tokens; it would require calibrated acceptance evidence, not just correctness
of the sparse matmul.

## Shader Timeline

A Metal System Trace with Shader Timeline enabled recorded a 256-token raw
prompt run. Filtering the shader list to the `dflash` process attributed
4,950.992 ms of sampled GPU work:

| Family | Attributed milliseconds |
| --- | ---: |
| MXFP4 target matmuls | 3,391.422 |
| BF16 GEMV/GEMM | 747.347 |
| Other target/runtime kernels | 361.149 |
| DFlash affine verify kernels | 146.861 |
| RoPE | 93.753 |
| Affine draft matmuls | 85.890 |
| Norm | 51.540 |
| Gated-delta kernels | 45.032 |
| Attention | 27.998 |

The top target kernel was
`mxfp4_qmv_wide_bfloat16_t_gs_32_b_4_nv_4_kl_16_batch_0`, with 2,408.455 ms
attributed and 83.083 us median interval. The block-five variant followed at
397.832 ms and 105.125 us median. This confirms native MXFP4 QMV as the main
target-side compute path.

Trace and summarized receipt:

- `build/reports/dflash/shader-profile/dflash-mxfp4-256.trace`;
- `build/reports/dflash/shader-profile/dflash-mxfp4-256-kernels.json`.

## QMV lane diagnostic

An experimental local MLX branch added BF16 MXFP4 QMV-wide instantiations and
an opt-in `MLX_MXFP4_QMV_WIDE_KL` override. Branch:
`work/mxfp4-qmv-wide-kl`, commit `85c43cea5`.

Focused validation:

- `PYTHONPATH="$PWD/python" .venv/bin/python -m unittest discover -s python/tests -p 'test_quantized.py'`
  passed 53 tests with one skip.
- KL8 and KL32 outputs were bitwise equal to KL16 on real M=4 and M=5 MXFP4
  linear modules.

Representative `gate_proj` timings, N=17408 and K=5120:

| M | KL16 | KL8 | KL32 |
| ---: | ---: | ---: | ---: |
| 4 | 0.281 ms | 0.308 ms | 0.408 ms |
| 5 | 0.331 ms | 0.362 ms | 0.455 ms |

Representative `down_proj` timings, N=5120 and K=17408:

| M | KL16 | KL8 | KL32 |
| ---: | ---: | ---: | ---: |
| 4 | 0.335 ms | 0.362 ms | 0.389 ms |
| 5 | 0.379 ms | 0.425 ms | 0.460 ms |

KL8 and KL32 are rejected. The branch remains local as a diagnostic and was
not promoted to MLX main.

## Strix

Strix was live during this pass:

- RTX 5070 Laptop GPU, 8,151 MiB total;
- 2 MiB used and 0% utilization at probe;
- 3.6 TB filesystem with 2.7 TB available.

No DFlash workload was run there because the useful runtime contains
Apple-Silicon-specific Metal kernels. Strix remains suitable for bounded CUDA
calibration work.

## Non-claims

- No representative multi-prompt quality or throughput claim is made.
- No sparse-head path was promoted.
- No QMV lane change was promoted.
- No DFlash2 MXFP4 quantization was performed; DFlash2 uses affine W4.
- The target remains the calibrated native MXFP4 artifact.
