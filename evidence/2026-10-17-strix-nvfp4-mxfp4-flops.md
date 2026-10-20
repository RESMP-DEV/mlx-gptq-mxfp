# Strix NVFP4/MXFP4 FLOP report

**Recorded:** 2026-10-17 14:50 America/Los_Angeles  
**Agent:** Codex  
**Host:** SSH alias `strix` (LAN `192.168.1.182`)  
**Source revision:** `mlx-gptq` `7131c407af8b8991fb5a8148a49cf2f323c7afd3`, dirty before this work  
**Scope:** live Strix inventory, native FP4 GEMM capability probes, clean NVFP4 cube and DFlash2-shaped FLOP measurements, and the MXFP4 software-support boundary. No model artifact was packed, run end-to-end, or accepted.

> **Correction at 15:01:** the initial “MXFP4 unavailable” conclusion was scoped too narrowly to
> PyTorch `scaled_mm` and cuBLASLt. Triton 3.8 executes native SM120 MXFP4 through
> `mma.sync...kind::mxf4nvf4...e2m1.e2m1...ue8m0`, numerical validation passes, and the corrected
> matched-path measurements are in the final section. The corrected best MXFP4 measurement is
> **312.912 TFLOP/s median**, not “unavailable”.

## Verified host state

The host answers SSH on the LAN and identifies itself as hostname `compute`; this is the machine configured locally as alias `strix`, not the separate B550/compute 3090 Ti host. It runs Ubuntu 26.04.1 with Linux `7.0.0-38-generic`, an Intel Ultra 9 275HX (24 CPUs), 91 GiB RAM, and a 3.6 TiB root filesystem with 2.7 TiB available.

The GPU is an NVIDIA GeForce RTX 5070 Laptop GPU, UUID `GPU-b8dfe31e-0e7e-ba02-cd75-bb2c39d58542`, CUDA capability `(12,0)`, 36 SMs, 8,151 MiB total memory, driver `595.99.02`, and NVSMI-reported CUDA 13.2. Maximum clocks are 3,090 MHz SM, 12,001 MHz memory. The active power limit was 100 W (minimum 5 W, default 50 W, maximum 115 W). While idle, PCIe negotiated Gen1/x8; advertised link maximum was Gen5/x16. NVIDIA did not expose a serial over NVSMI (`[N/A]`).

The benchmark environment was the `/home/kearm/mlx-dynamic-mxfp4/.venv` interpreter with PyTorch `2.14.1+cu130` and CUDA runtime 13.0. The host separately has CUDA toolkit/nvcc 13.4 (`V13.4.92`). This version difference is recorded rather than hidden.

## In-flight calibration outcome

At the start of inspection, PID 10030 was running a detached tmux calibration:

```text
python -m mlx_gptq.dflash2_calibrate \
  --model /home/kearm/models/z-lab-Qwen3.8-27B-DFlash2 \
  --dflash-package /home/kearm/dflash-pytorch \
  --capture-dir /home/kearm/mlx-gptq-dflash2/capture-full \
  --output /home/kearm/mlx-gptq-dflash2/calib-full-mxfp4 \
  --device cuda:0 --cpu-device cpu \
  --cpu-solve-min-in-features 8192
```

It completed normally after 789.613 seconds. Its receipt records 1,981 cycles, 47 tensors, median relative error `0.118944`, and maximum relative error `0.176118`. The calibration's own non-claims remain authoritative: exact projected contexts and mask-block embeddings were replayed, the two selector codebooks remained BF16, and no runtime target-agreement improvement is claimed.

The first NVFP4 timing attempt ran while this calibration held approximately 7.05 GiB during CPU-heavy solve phases. That run is retained separately as `results-coresident.json`; after calibration exited, the clean result below became the primary result. The co-resident 6144-cube median was within about 1% (314.78 versus 317.62 TFLOP/s), while the launch-dominated 1024-cube median was materially lower (52.29 versus 90.69 TFLOP/s).

## Method

The benchmark calls the native `torch.nn.functional.scaled_mm` / `torch._scaled_mm_v2` path. Canonical two-level NVFP4 uses:

- operands: `torch.float4_e2m1fn_x2`;
- block scales: `torch.float8_e4m3fn`, `BlockWise1x16`, `SWIZZLE_32_4_4`;
- global scales: `torch.float32`, `TensorWise`, `NO_SWIZZLE`;
- output: BF16;
- operation count: `2*M*N*K`;
- timing: CUDA events after 4 warmups, 16 measured calls;
- quantization exclusion: operands are valid random E2M1x2 bytes and scale values are one; packing/quantization cost is not included.

MXFP4 attempts use `torch.float8_e8m0fnu`, `BlockWise1x32`, and `SWIZZLE_32_4_4`. Every attempted cube and workload shape raises the same PyTorch error:

```text
NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300
```

A separate CUDA 13.4 cuBLASLt heuristic-only C++ probe was compiled and run against the native library. It requested FP4/BF16 GEMM algorithms for both ABI-exposed scale modes:

```text
NVFP4_VEC16_UE4M3 status=15 ... algorithms=0 result_state=7
MXFP4_VEC32_UE8M0 status=7  ... algorithms=0
```

`status=15` is `CUBLAS_STATUS_NOT_SUPPORTED`; `status=7` is `CUBLAS_STATUS_INVALID_VALUE` for that exact probe configuration. Therefore PyTorch's working NVFP4 implementation is not coming from this tested cuBLASLt heuristic configuration. The result strengthens the MXFP4 software boundary but is configuration-specific and is not proof that RTX 5070 silicon lacks MXFP4 support in every possible stack.

## Clean NVFP4 cube results

No CUDA compute application was active during this run.

| M=N=K | p50 time (µs) | p90 time (µs) | best time (µs) | p50 TFLOP/s | best TFLOP/s |
---:|---:|---:|---:|---:|---:|
| 1024 | 23.680 | 28.960 | 20.096 | 90.688 | 106.861 |
| 2048 | 71.328 | 73.248 | 71.168 | 240.857 | 241.399 |
| 4096 | 450.928 | 455.296 | 448.768 | 304.791 | 306.258 |
| 6144 | 1,460.416 | 1,547.520 | 1,449.312 | 317.619 | 320.053 |
| 8192 | 3,641.408 | 3,657.248 | 3,567.520 | 301.947 | 308.201 |

The clean-run monitor sampled SM clocks up to 1,425 MHz, power up to 84.68 W, utilization up to 35%, and up to 606 MiB memory use. Low utilization samples are expected for sub-millisecond kernels launched in short bursts by this event-per-call harness.

## DFlash2-shaped NVFP4 results

These cases use the real projection dimensions observed in the calibration manifest: Q approximately `(K=5120,N=4096)`, gate/up `(K=5120,N=17408)`, and down `(K=17408,N=5120)`.

| M | K | N | p50 time (µs) | p90 time (µs) | p50 TFLOP/s |
---:|---:|---:|---:|---:|---:|
| 128 | 5,120 | 4,096 | 26.336 | 31.360 | 203.854 |
| 128 | 5,120 | 17,408 | 158.160 | 159.456 | 144.265 |
| 128 | 17,408 | 5,120 | 195.040 | 196.224 | 116.986 |
| 8,192 | 5,120 | 4,096 | 1,116.736 | 1,158.304 | 307.680 |
| 8,192 | 5,120 | 17,408 | 4,963.840 | 4,968.960 | 294.185 |
| 8,192 | 17,408 | 5,120 | 4,800.128 | 4,806.784 | 304.219 |

The workload monitor observed utilization up to 99%, power up to 92.69 W, SM clock up to 1,425 MHz, and memory up to 890 MiB.

## Initial interpretation, superseded by the Triton result

- Measured clean NVFP4 GEMM peaks at **317.6 TFLOP/s median** on the 6144 cube and sustains **301.9 TFLOP/s median** at 8192.
- The model-like 8,192-row projections sustain **294.2 to 307.7 TFLOP/s median**; the M=128 cases are 116.99 to 203.85 TFLOP/s and are not a peak-throughput measurement.
- PyTorch `scaled_mm` rejects MXFP4 as B200/B300-only, and the direct CUDA 13.4 cuBLASLt probe found no algorithm for the tested MXFP4 configuration.
- That established a library-route boundary only. It did not establish an SM120 hardware boundary; the Triton result below supersedes this interpretation.

## Non-claims

- The initial library-only interpretation is retained as history and is not the final capability verdict.
- It does not measure quantization/packing overhead, activation load overhead, model prefill/decode speed, accuracy, perplexity, target agreement, or a packed DFlash2 artifact.
- Random E2M1 bytes and unit scales are suitable for tensor-core throughput probing, not for numerical-quality ranking.
- The power/clock figures are sampled telemetry, not a locked-clock thermal sweep. The laptop's power and thermal ceiling may differ under a sustained workload.
- No manufacturer peak or sparse/dense topology claim is inferred from these measurements.

## Artifacts and checks

Local raw artifacts live in ignored `build/reports/strix-nvfp4-mxfp4-20261017/`; the mirrored remote directory is `/home/kearm/reports/strix-nvfp4-mxfp4-20261017/`.

| Artifact | SHA-256 |
|---|---|
| `bench.py` | `a7c389e9ab143b477de3bfd14c016668dc9a0cdb3488c2b87352047a9b8ded64` |
| `cublaslt_probe.cpp` | `704a2e6deff78937ce8f85741ba2babdf4e5b742bc1b3f656a6235c41c63da8e` |
| `results.json` | `072e742da2d5d7d93f847ea1bcf6f6695361f75a88622b2601fc32034ba52a96` |
| `results-coresident.json` | `9f37d04c07e293981a1a62255f6db0d979e1ffc947c9378b7a6ccae4f166ed0c` |
| `results-workload.json` | `fa9820260b01efa7e02385702ca501fd945b7eeaad8fb9933a9c3251586f076f` |
| `monitor-clean.csv` | `0eb419f1fbeccd729842c71441a490717cd9761aa5fbc97ad507561dfcb8a5b6` |
| `monitor-workload.csv` | `9c503cdd3ba805cb1ebe121cf1cc41cc2ada1a51ef2c62bf400d7d63495d4e9e` |
| `monitor-coresident.csv` | `ae03a59ee00b91e1d8e5a8d138ad86abd7d138597b1f055e75001242ff317b2e` |
| `inventory.txt` | `e0ed2e9797f3134d0f5e916d854f32703aaa301ac6c8cba32b06cb36896b65a6` |
| `cublaslt-probe.txt` | `364189ca218819cee6eef62f265c653fa06249fe6d9a42e9d9105f22d8d92f55` |
| `calibration-receipt.json` | `1e097d8d6306c540d2b0ac801824cd926633b841ca008e26c713149bbeea7c5e` |

Validation that actually ran:

- SSH identity, GPU, CPU, memory, disk, PCIe, clocks, and power inventory: exit 0.
- `uvx ruff check build/reports/strix-nvfp4-mxfp4-20261017/bench.py`: exit 0, all checks passed.
- Clean cube NVFP4 benchmark: exit 0; all outputs finite and all MXFP4 attempts rejected.
- DFlash2-shaped NVFP4 benchmark: exit 0; all outputs finite and all MXFP4 attempts rejected.
- CUDA 13.4 probe compile/link and execution: exit 0, with the unsupported/invalid heuristic statuses recorded above.
- Full DFlash2 MXFP4 calibration completion receipt: exit 0 in its detached tmux job.

The pre-existing dirty files in `mlx-gptq` were not modified. This report, ignored benchmark artifacts, and the copied remote report directory are the task-owned additions.

## Correction: native SM120 MXFP4 through Triton

**Recorded:** 2026-10-17 15:01 America/Los_Angeles.

The user correctly challenged the initial conclusion. The decisive route is Triton 3.8
`tl.dot_scaled(lhs, lhs_e8m0_scale, "e2m1", rhs.T, rhs_e8m0_scale, "e2m1")`. A small
non-square numerical test at `M=128, N=256, K=512` passed Triton's reference check for both
MXFP4 and NVFP4 after grouped rasterization.

The final generated PTX contains this exact instruction:

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
```

This is a native SM120 block-scaled MXFP4 tensor-core operation, not BF16 software emulation.
The absence of `tcgen05` is expected on this consumer-Blackwell path; SM120 uses the
`mma.sync` block-scaled instruction family.

The upstream Triton block-scaled tutorial was adapted only where necessary for SM120:

- removed an import absent from installed Triton 3.8;
- removed an invalid tutorial `clc` launch keyword;
- reduced the tile from the SM100 shared-memory requirement to SM120's approximately 99 KiB limit;
- added grouped rasterization and tested two and three pipeline stages;
- passed tile configuration explicitly into the benchmark.

Tile search found `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128`, four warps, and grouped
rasterization as the best general configuration tried. At 8192³, grouped rasterization plus
three stages improved MXFP4 from `184.675` to `289.487 TFLOP/s` median; the dedicated tile
search observed `294.737 TFLOP/s`. This confirms the stage-two 8192³ drop was scheduling, not
an MXFP4 format ceiling.

## Corrected matched Triton results

Both formats use the same Triton kernel family, `128×128×128` tile, four warps, grouped
rasterization, three stages, FP16 output, four warmups, and sixteen measured calls.

| M | K | N | MXFP4 p50 TFLOP/s | NVFP4 p50 TFLOP/s | MX/NV |
---:|---:|---:|---:|---:|---:|
| 1,024 | 1,024 | 1,024 | 101.373 | 101.068 | 1.0030 |
| 2,048 | 2,048 | 2,048 | 255.531 | 255.470 | 1.0002 |
| 4,096 | 4,096 | 4,096 | 302.186 | 297.209 | 1.0167 |
| 6,144 | 6,144 | 6,144 | 283.171 | 279.704 | 1.0124 |
| 8,192 | 8,192 | 8,192 | 289.487 | 279.285 | 1.0365 |
| 128 | 5,120 | 4,096 | 155.417 | 156.504 | 0.9931 |
| 128 | 5,120 | 17,408 | 147.047 | 141.432 | 1.0397 |
| 128 | 17,408 | 5,120 | 128.393 | 125.412 | 1.0238 |
| 8,192 | 5,120 | 4,096 | 312.912 | 306.560 | 1.0207 |
| 8,192 | 5,120 | 17,408 | 281.245 | 273.623 | 1.0279 |
| 8,192 | 17,408 | 5,120 | 293.652 | 286.918 | 1.0235 |

Corrected conclusions:

- Native SM120 MXFP4 is available and reaches **312.912 TFLOP/s median** on the DFlash2 Q-projection shape `8192×5120×4096`.
- On the identical Triton path, MXFP4 is **0.7% to 4.0% faster** than NVFP4 on all tested shapes except `128×5120×4096`, where it is 0.7% slower.
- The best available framework paths are also close: PyTorch/cuBLAS NVFP4 reached **317.619 TFLOP/s median** at 6144³, while native Triton MXFP4 reached **312.912 TFLOP/s median**. The 1.5% gap is across different backends and should not be interpreted as a format-level deficit.
- The appropriate dense-FLOP inference is now direct and native: MXFP4 and NVFP4 have effectively the same FP4 tensor-core throughput on this RTX 5070 Laptop GPU, with MXFP4 modestly faster in the matched Triton kernel.

The monitor for tuning and matched runs sampled utilization up to 100%, power up to 95.11 W,
SM clocks up to 2,752 MHz, and memory up to 7,645 MiB. The memory peak includes test-data
generation and tile search, not just kernel operands.

## Additional correction artifacts

| Artifact | SHA-256 |
|---|---|
| `triton_block_scaled_matmul.py` | `c554ec603b336d078498bf2a866a49cfdfc4a1f969c5cebe5318a9b926bfb946` |
| `triton_fp4_bench.py` | `6b8ffbe563726acc09b029662d93ecd861aa5888dbd099305812b926503b675b` |
| `triton_mxfp4_tune.py` | `187cf65b3be127d59260e19dce5501b82452f3d9db6f919082f1a1bfe43f86fe` |
| `triton_sm120_mxfp4_probe.py` | `8bbb4d86f332422722733ec0652ce54ccbca8603fbb98a7c5faa722fc0fd4cf3` |
| `results-mxfp4-triton.json` | `41c6074a20ddf08a137b0d7c693443f32744a63637e05c7c02ed108263920f61` |
| `results-nvfp4-triton.json` | `459be25c8965e7be7a18e4fc8d9e022a8b40d6c4232277b38f813a6756c6856d` |
| `results-mxfp4-triton-ungrouped-s2.json` | `a22abbe90e4b1997f307803f8278492aeb4ed4e1666a6ec59d558f8e0482dbab` |
| `results-nvfp4-triton-ungrouped-s2.json` | `4cddec2cf7c09bab469b2f647e0e3484a95f9bb394a45c1fb9f1a471c35df82a` |
| `triton-mxfp4-tune.json` | `23c09ca88f4b54e8b2a1d565e85d2fce1d91524580ae62b360f174a78a35985f` |
| `triton-mxfp4-tune-8192.json` | `7125808383d5e363e54940be2d26bafab6b4adcef2fc189ba0a4a71c06ebd559` |
| `monitor-triton-fp4.csv` | `13c0be5f667215e89ec452c99b04d9c01c0005f80e78c74d2a298870e158bd64` |
| `triton-mxfp4--922196040223698101.ptx` | `926f223766aacf88a2a818045a19b875eb55f9b7c161b98c3a2b36a5875c2435` |
| `triton-mxfp4--922196040223698101.cubin` | `b8656498ed9fe11ad402ab9e56c26624ff81b22849c7c6129486be78086ee0cf` |

Additional validation that actually ran:

- `uvx ruff check .../triton_mxfp4_tune.py`: exit 0.
- `uvx ruff check .../triton_fp4_bench.py`: exit 0.
- Final grouped MXFP4 numerical validation: exit 0.
- Final grouped NVFP4 numerical validation: exit 0.
- Native MXFP4 and NVFP4 matched benchmarks: exit 0; all sampled outputs finite.
- PTX inspection found the exact `e2m1.e2m1...ue8m0` native MXFP4 MMA instruction in the final kernel.

## Why the first pass missed native MXFP4

**Recorded:** 2026-10-17 15:05 America/Los_Angeles.

A post-correction search found no previous durable record of Triton MXFP4 on this machine:

- Codex memory and rollout summaries: the only relevant Strix entries identify the RTX 5070 and its Ubuntu/NVIDIA bring-up; they do not mention `dot_scaled`, `mxf4nvf4`, Triton MXFP4, or native SM120 FP4 throughput.
- Shared-memory index/full-text search: no relevant hit.
- Existing project documentation: older Strix records cover CUDA calibration and Stage A work, not this GEMM route. The first project hit is this corrected report.

The route was generically present but not discoverable as an SM120 capability claim:

- Installed Triton 3.8 documents `tl.dot_scaled` and E2M1/E8M0 inputs, but its API text says unsupported architectures use BF16 software emulation. It is not a support matrix for SM120.
- The upstream block-scaled tutorial says hardware acceleration is for NVIDIA compute capability 10 and its `supports_block_scaling()` gate accepts CUDA major versions `[10, 11]`. RTX 5070 is capability `(12,0)`, so the stock tutorial prints “requires GPU support” and exits before compilation.
- Native SM120 support became visible only after bypassing that gate, reducing the tile for SM120 shared memory, correcting tutorial drift, compiling, and inspecting PTX for `kind::mxf4nvf4...ue8m0`.

The first investigation therefore stopped too early at two library routes, PyTorch `scaled_mm` and a cuBLASLt heuristic probe, and generalized their “no MXFP4 algorithm” result too broadly. Its initial filesystem search also looked for NVFP4/MXFP4 artifact names rather than installed compiler APIs or block-scaled GEMM operations. The report had the correct narrow non-claim (“not proof silicon lacks MXFP4”), but its headline interpretation “MXFP4 unavailable” exceeded the evidence.

Durable lesson: for a hardware capability claim, enumerate and probe at least the major execution engines separately: the framework BLAS route, Triton, CUTLASS/CuTe, and direct PTX/SASS. A library rejection is evidence about that library and configuration, not a substitute for compiler lowering or silicon-level proof.
