# Receipt: Strata-inspired MXFP4/Optane MLX architecture

**Date:** 2026-10-05
**Agent:** Codex parent (GLM), four timed-out implementation workers, two completed reviewer workers, focused parent implementation
**Worktree:** `/Users/kearm/mlx/.worktrees/strata-mxfp4-optane`
**Branch:** `research/strata-mxfp4-optane`
**Base:** MLX `af219595b289cea24eb01dd7dd915bcc785fe1b7`

## Scope

Created the implementation architecture and first native-MXFP4 external PLE reader/benchmark, then implemented and cross-machine validated dynamic native-MXFP4 orchestration for the tiny qwen4-exp path. Preserved the prior streaming-loader/OOM evidence and the existing qwen4-exp parity work. No push, PR, issue comment, upstream change, or B550 quantization launch occurred.

## Outputs

- Architecture: `docs/architecture/strata-mxfp4-optane.md`
  SHA-256 `820e42891244951a6cf78fdbafae6ad07924c72b66a1c0c1cef9d44fd4c05b93`
- MXFP4 benchmark: `benchmarks/python/strata_mxfp4_ple_bench.py`
  SHA-256 `169ff7b6c6b29a0e70314a81e2f09f08911f1e08174e0f396147b8d5d5de0250`
- Seeded 90-byte control: `benchmarks/python/storage_table_bench.py`
  SHA-256 `06ffc747ab81acc4419a64afc88559a601ac8482df60a9505cf15424325ca4fd`

## Architecture decisions

- Resident backbone uses standard MLX MXFP4, group size 32, bits 4.
- A 160-value PLE row is `uint32[20]` packed codes plus `uint8[5]` E8M0 scales: 80 + 5 bytes.
- The first external format uses separate codes/scales planes and native `mx.dequantize`.
- The production reader must use bounded admission across both planes; worker count alone is not a queue bound.
- The first artifact omits MTP and vision. MTP becomes a separate indexed draft artifact in M6.
- Stage A runs on B550 under a cgroup with `MemoryMax=28G`, `MemorySwapMax=0`, `TasksMax=256`, and `MAX_JOBS=2`.
- Stage B streams headers and shards on the Mac. The stock lazy BF16 loader is forbidden.

The architecture document records the full milestone and acceptance gates.

## Live MXFP4 Optane result

Current device: Intel `ET000750KWJTF`, 750.16 GB, ExFAT `/Volumes/OPTANE`, mounted as `disk12s2`.

Synthetic full-size files:

- `/Volumes/OPTANE/mlx-storage-table-bench/mxfp4-ple-full.codes`
  25,600,122,880 bytes
- `/Volumes/OPTANE/mlx-storage-table-bench/mxfp4-ple-full.scales`
  1,600,007,680 bytes
- logical rows: 320,001,536

Hardened fresh-index no-cache run:

```text
tokens/run: 256
rows/token: 16
warmups: 1
measured runs: 5
max inflight workers: 256
max selected rows/ticket: 65536
page reads: 40487
storage bytes: 664661504
storage throughput: 1089987614.96 B/s
logical row throughput: 2854762.05 B/s
synthetic tokens/s: 2099.09
dequantize time: 0.092168042 s
Metal peak: 4292608 bytes
```

Receipt: `build/reports/strata-mxfp4-workers/mxfp4-ple-optane750-full-hardened.json`.

Full-size cache arm:

- capacity: 1,048,576 rows
- filled: 4,099 rows
- measured cache hits: 12,288
- measured page reads and storage bytes: 0
- synthetic tokens/s: 15,392.94

Receipt: `build/reports/strata-mxfp4-workers/mxfp4-ple-optane750-full-cache.json`.

## Validation

Existing qwen4-exp tiny-reference parity:

```text
PYTHONPATH=/Users/kearm/mlx-lm \
  /Users/kearm/mlx/.venv/bin/python research/qwen38-flash/verify/dump_mlx_logits.py
PYTHONPATH=/Users/kearm/mlx-lm \
  /Users/kearm/mlx/.venv/bin/python research/qwen38-flash/verify/compare.py --tol 2e-4
```

All six tiny variants passed all dense, sparse, and EOS-reset prompts. Maximum absolute logit difference: `1.192e-07`. Receipt: `build/reports/strata-mxfp4-workers/qwen4-exp-tiny-parity.log`.

```text
ruff check --fix benchmarks/python/strata_mxfp4_ple_bench.py
ruff format benchmarks/python/strata_mxfp4_ple_bench.py
ruff check benchmarks/python/strata_mxfp4_ple_bench.py
```

Exit 0.

The benchmark verifies first, middle, and last rows against deterministic `mx.quantize(..., mode="mxfp4")` plus `mx.dequantize`. The hardened reader validates row bounds, integer indices, row count, cache size, selected-row ticket size, partial plane pairs, short reads, EOF, thread-safe cache updates, and cancels remaining futures on collection failure.

Whitespace checks passed for the architecture and benchmark.

## B550 preflight

Live read-only receipt: `build/reports/strata-mxfp4-workers/b550-preflight.json`.

- hostname: `compute`
- kernel: `7.0.0-34-generic`
- GPU 0: RTX 3090 Ti, 7,128,219,648 bytes used, 5% utilization
- GPU 1: RTX 3090 Ti, 34,603,008 bytes used, 0% utilization
- host memory available: 80,538,001,408 bytes
- swap: 8,589,930,496 used of 8,589,930,496; free is zero
- home disk free: 3,167,916,322,816 bytes
- active GPU process: one `salm-lora` evaluation using about 6.8 GiB

Stage A remains unlaunched because swap is fully committed and another GPU workload is active.

## Worker outcomes

The first four broad workers (`architecture`, `mxfp4-bench`, `gptq-oom-plan`, `mlx-lm-integration`) timed out at 900 seconds with exit 124 and produced no final edits. Their status/event receipts remain under `build/reports/strata-mxfp4-workers/` and private CCR run directories; they are not acceptance evidence.

Two focused reviewers completed successfully on FlashX:

- architecture review: `build/reports/strata-mxfp4-workers/architecture-review.txt`
- benchmark review: `build/reports/strata-mxfp4-workers/mxfp4-bench-review.txt`

The parent corrected the reviewer-found `uint32[5]` documentation error to `uint32[20]`, fixed byte totals, tightened M1/M2/M4 gates, clarified cache mutability and MTP scope, and hardened benchmark validation and failure handling.

## Non-claims

- This is not a full model artifact and contains synthetic PLE values.
- No real Qwen token trace, model weights, logits, perplexity, or generation output was run.
- `F_NOCACHE` was requested, but macOS cache behavior was not independently traced.
- Two-plane storage is slower than the 90-byte control because codes and scales may require separate pages.
- No B550 GPTQ Stage A smoke or cgroup execution ran.
- No CUDA kernel, MTP path, fused Metal PLE kernel, or expert cache was implemented.
- The current benchmark caps selected rows per ticket; the production bounded-admission generations remain to be implemented.

## Later recheck

The user challenged the missing durable-model step. Parent rechecking established:

- Prior MXFP4 GPTQ packing is real and durable for LFM2.5, but not FlashNext.
- The local FlashNext community artifact is 103.86 GiB and affine. It is a diagnostic real-checkpoint control, not the target quantization.
- The official 359,999,963,128-byte BF16 source is absent from the Mac cache.
- The quantizer environment uses Transformers 5.14.1; qwen4_exp requires 5.17.0.
- Existing Stage A materializes all parameters in a layer; real FlashNext layer 1 includes the complete BF16 PLE table and cannot fit.
- Strix is live with an idle 8,151 MiB RTX 5070, but has no ML environment, quantizer checkout, or model cache.
- B550 has two effectively idle 24 GB GPUs, 76 GiB available memory, and all swap committed.
- A focused worker found the real snapshot uses interleaved MRoPE while the current MLX-LM port and tiny generator do not cover it. This is an additional correctness blocker.

Focused reports:

- `build/reports/strata-mxfp4-workers/recheck-qwen4exp-model-state.txt`
- `build/reports/strata-mxfp4-workers/recheck-strix-calibration-slice.txt`

The broad quantizer-compatibility worker timed out again and is retained only as a failure receipt at `build/reports/strata-mxfp4-workers/recheck-quantizer-compat.txt.status.json`.

## Dynamic MXFP4 implementation

The dynamic quantizer is now mode-aware rather than affine-only. `--mxfp4`
selects native MXFP4 for the baseline and native MXFP8 for sensitive layers. The
implementation rejects invalid native bit/group combinations, skips leaves whose
matrix alignment cannot satisfy either selected mode, keeps embeddings out of
the gradient-calibration set, and preserves per-layer mode overrides in the saved
MLX config.

The full tiny qwen4-exp CLI exposed two additional model-path requirements. QSA
block indices and MoE expert indices are now gradient-stopped, preserving forward
indices while making sensitivity calibration possible. N-gram IDs are computed
with exact host-side int64 arithmetic because MLX 0.32.3 on Linux CPU cannot
evaluate that integer graph; the embedding and neural tensors remain MLX arrays.

Local and Strix receipts:

- Implementation worktree: `/Users/kearm/mlx-lm/.worktrees/dynamic-mxfp4`
- Branch/base: `dynamic-mxfp4` from `e396b18b87e6274822c05bd6af14e223e9fbfce6`
- Final patch SHA-256:
  `80a9b1c83fb3cda96ccdc69f6b18bce2ca3c61f2fd0ae78636648ae9d3a760b3`
- Strix copy: `/home/kearm/mlx-dynamic-mxfp4`
- Consolidated receipt:
  `build/reports/dynamic-mxfp4/receipt.json`

Mac validation:

```text
PYTHONPATH=$PWD /Users/kearm/mlx/.venv/bin/python -m unittest \
  tests.test_models.TestModels.test_qwen4_exp -v
PYTHONPATH=$PWD /Users/kearm/mlx/.venv/bin/python -m unittest discover \
  -s tests -p test_dynamic_quant.py -v
```

Both exited 0; five dynamic tests passed. The complete CLI over the bounded
10,000-ID tiny fixture exited 0 in 12.76 seconds with 0.627 GiB MLX peak and
340,393,984 bytes maximum RSS. The saved model reloaded with finite float32
logits and contained 59 MXFP4 modules plus one MXFP8 override.

The existing BF16 tiny-reference parity was rerun against the modified model.
Dense, sparse, and EOS-reset prompts all passed with a maximum absolute logit
difference of `1.043e-07`, below the `2e-4` gate. Receipt:
`build/reports/dynamic-mxfp4/mac/tiny-parity.log`.

Strix validation:

```text
systemd-run --user --scope --collect -p MemoryMax=2G \
  ../.venv/bin/python -m pytest -q tests/test_dynamic_quant.py
systemd-run --user --scope --collect -p MemoryMax=3G \
  /usr/bin/time -v ~/mlx-dynamic-mxfp4/.venv/bin/mlx_lm.dynamic_quant \
  --model ~/mlx-dynamic-mxfp4/fixtures/tiny-qwen4exp-cli \
  --mlx-path .../mlx_model --mxfp4 --target-bpw 5.0
```

The five tests passed. The CLI exited 0 in 160.28 seconds under the 3 GiB
cgroup with 0.815 GiB MLX peak, 1,724,260 KiB maximum RSS, and zero swaps. Its
saved model reloaded with the same 59 MXFP4 plus one MXFP8 selection and finite
float32 logits. The Mac and Strix quantization configs are SHA-256 identical;
packed-weight hashes differ across Metal and Linux CPU, so packed bytes are not
claimed bit-identical.

The complete upstream `test_qwen4_exp` harness still fails on Strix in its lazy
float16 path with `GatherMM::eval` only supporting float32. The float32 CLI and
reload pass, so this is recorded as an MLX Linux CPU backend limitation, not a
dynamic-MXFP4 acceptance failure.

## Tiny campaign-owned GPTQ-MXFP4 artifact

The dirty GPTQ main checkout was preserved. A clean isolated worktree was
created from `7131c407af8b8991fb5a8148a49cf2f323c7afd3` at
`/Users/kearm/research/ml/RESMP-DEV/mlx-gptq-mxfp/.worktrees/qwen4-exp`.
It now requires Transformers 5.17 or newer, makes Accelerate explicit, and moves
concrete buffers to the selected forward device while leaving meta parameters
untouched. Final patch SHA-256:
`401501e0b6621628f096047a16a1dd40b1c4ceef3d1cc18b89d3cf2a3382d25d`.

Stage A used a fused-PLE HF-format tiny control and excluded all PLE, router,
gates, and shared-expert gates. Mac CPU completed all four layers in 2.32
seconds, emitted 144 tensor records, used 357,941,248 bytes maximum RSS, and
swapped zero times. No PLE tensor entered the artifacts.

Stage B converted from the matching sharded-PLE MLX fixture and injected all
144 calibration records into 60 packed MLX tensors. One global mixer tensor
stayed RTN because it is outside the decoder-layer artifact scope. The resulting
standard MLX checkpoint has native MXFP4/group 32 configuration, a 2,915,507-byte
weight file, and reloads/generates successfully. Model SHA-256:
`751758990f4af3813e34006b28c9cb3a53f69c5e4f6d041aefffc620775c36cd`.

The same GPTQ source and fused fixture were copied to Strix. After the buffer
fix, the RTX 5070 completed all four layers in 4.97 seconds under a 3 GiB
cgroup, using 2,526,704 KiB maximum RSS and a sampled 290 MiB GPU peak, with
zero swaps. The durable tiny artifact was copied back to Strix and reloaded with
finite logits and 61 MXFP4 modules.

Mac and RTX 5070 Stage A layer hashes were identical for layers 0 and 1 and
differed for layers 2 and 3, consistent with backend reduction order. The
durable Mac Stage B artifact consumed the Mac CPU artifacts.

Receipts:

- `build/reports/gptq-qwen4-exp-tiny/receipt.json`
- `build/reports/gptq-qwen4-exp-tiny/mac/stage-a.log`
- `build/reports/gptq-qwen4-exp-tiny/mac/stage-b.log`
- `build/reports/gptq-qwen4-exp-tiny/strix/stage-a.log`
- `build/reports/gptq-qwen4-exp-tiny/strix/gpu-memory.csv`

This is the first complete campaign-owned GPTQ-MXFP4 Stage A/B execution for
the qwen4-exp shape, but it is only a tiny resident-PLE control. It does not
prove official-model quality or solve real PLE splitting.

## BF16 source transfer

The official source is now being pulled to both execution machines from the
immutable Hub revision:

- Repository: `Qwen/Qwen3.8-Flash-Next`
- Revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Expected tree: 144 files, 360,023,351,514 bytes

Local path and controller:

- `/Users/kearm/models/Qwen3.8-Flash-Next-BF16`
- detached tmux session `qwen-bf16`
- explicit `/Users/kearm/toolbelt/bin/hfd.sh` wrapper, hfd 0.1.1 + aria2c,
  12 jobs, 12 connections per file
- configured full hash verification

Strix path and controller:

- `/home/kearm/models/Qwen3.8-Flash-Next-BF16`
- detached tmux session `qwen-bf16`
- explicit `/home/kearm/mlx-dynamic-mxfp4/reports/bf16-transfer/hfd.sh`
  wrapper, hfd 0.1.1 + aria2c, 12 jobs, 12 connections per file
- configured full hash verification

Both transfers resumed from partial files and were alive in detached tmux at the
receipt time. A 60-second sample measured 3.98 MB/s on Mac and 7.08 MB/s on
Strix. Strix initially used `hf download` with Xet, but its staging path stalled
with open sockets and no file growth; it was switched to the same pinned
hfd/aria2 route. The Mac launcher was also moved into tmux after its direct
nohup child was reaped at shell-session exit.

In-progress receipt:
`build/reports/bf16-transfer/receipt.json`.

The download and full hash verification are not complete yet, and no real-model
GPTQ calibration is claimed from this source yet.

## Real Stage A readiness update

The remaining fused-PLE OOM boundary is implemented. The GPTQ pipeline now has
`--lazy-ple`: it replaces Transformers' fused `320,001,536 x 160` PLE parameter
with selected-row reads from the official `shard_0` through `shard_127`
safetensors tensors. The real index places those shards across source files 5
through 37. Lazy PLE is excluded from quantization and never allocates the fused
table.

Final implementation patch SHA-256:
`f33c0c8ebe1dde33b3433e64d91082e6c81f29a9d4d939c65dc7e3bcb6973b5d`.
The isolated GPTQ tests passed 34/34.

The tiny sharded-source control completed on Mac with `--lazy-ple`, emitted zero
PLE artifacts, and produced all four layer artifacts bit-identical to the prior
fused-PLE control. It used 358,334,464 bytes maximum RSS.

The same control passed on Strix's RTX 5070 under a 3 GiB cgroup in 5.88
seconds with 290 MiB sampled GPU peak and zero swaps. B550's cloned
Transformers 5.17 / Torch CUDA 2.14.1 environment sees both RTX 3090 Ti GPUs;
its dual-GPU lazy-PLE run passed under a 16 GiB cgroup in 9.75 seconds with
zero swaps.

The real config probe on B550 constructs `Qwen4ExpForCausalLM` with 48 layers.
Layer 1 exposes the fused PLE shape as a meta parameter, and the real config
confirms interleaved MRoPE section `[11, 11, 10]`.

Calibration is staged from the pinned model-agnostic corpus at revision
`7d279eed2bdc6c4ea391a549af7999b7b4a20484`: 32 windows of 2,048 tokens,
SHA-256 `a8073b766fc6b3091336c6bbb982687eb1279709597823c94e079c64fbc069e5`.
Its tokenizer was copied from the pinned diagnostic conversion while the
official tokenizer file is still part of the pending transfer, so publication
tokens must be regenerated after source verification.

A `qwen-to-b550` tmux watcher now waits for the Strix download to finish, runs
full hash verification, and then transfers the verified tree directly to B550
over SSH on the LAN. Once that copy finishes, the watcher starts B550's
detached `qwen-stage-a` tmux session automatically with 32 by 2,048 calibration
tokens, native MXFP4, lazy PLE, both RTX 3090 Ti GPUs, an 18 GiB budget per
GPU, and a 68 GiB host-memory cgroup. Consolidated receipt:
`build/reports/gptq-real-readiness/receipt.json`.

Stage A is therefore ready to launch as soon as a verified source tree reaches
B550. Final-artifact work remains gated on streaming Stage B, external MXFP4
PLE planes, MRoPE parity, and real-model quality acceptance.

## PLE storage decision correction

The user challenged the assumption that PLE rows should be quantized. The
correct calibration rule is exact selected BF16 reads; `--lazy-ple` already
follows that rule and produces bit-identical tiny layer artifacts to the fused
BF16 control.

For final storage, native BF16-on-disk is now the default. A full PLE table is
102,400,491,520 bytes BF16 versus 27,200,130,560 bytes as native MXFP4 planes,
but both row representations fit within one 16 KiB page. Random PLE access may
therefore be page-traffic-bound, and the 3.765x logical-byte ratio must not be
assumed to become an end-to-end speedup.

The source BF16 shards must be retained. A later MXFP4 PLE conversion is allowed
only after a same-index Optane benchmark on a real trace proves lower end-to-end
latency and passes the PLE quality gate. A reusable row-width benchmark was
added at `benchmarks/python/ple_row_width_bench.py`; `/Volumes/OPTANE` is
currently unmounted, so the decisive external-storage A/B remains pending.

## Transfer correction

The low percentages quoted in conversation were stale. Live `hfd.sh` status and
a measured interval showed both transfers active. At the latest check Mac had
90.20 GiB written and measured 49.86 MB/s; Strix had 59.67 GiB written and
measured 11.64 MB/s. Both use explicit `hfd.sh` wrappers, pinned revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, aria2c resume, and configured full
verification. No HTTP 429 was present; prior "429" text was the byte value
`429MiB` in progress output.

Because B550 performs calibration and Strix was the slower pull, the Strix-to-
B550 relay was stopped. B550 now runs its own explicit `hfd.sh` pull in tmux
session `qwen-bf16`; after full verification it automatically starts the
existing `qwen-stage-a` controller.

## Full source landed and real Stage A began, 2026-10-07

Both authoritative execution copies completed and passed full hash
verification at pinned revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`:

- Mac: 144/144 files, 360,023,351,514 bytes, completed 2026-10-07T01:29:23Z.
- B550: 144/144 files, 360,023,351,514 bytes, completed 2026-10-07T01:46:29Z.

Mac had one transient aria2 failure on `model-00002`; B550 had one missing
`model-00009`. Both were repaired through explicit `hfd.sh` resume calls before
verification. Strix was unreachable by SSH and ICMP during the checks, so its
copy has no current completion claim.

Real GPTQ-MXFP4 Stage A is running on B550 in detached tmux session
`qwen-stage-a`, using both RTX 3090 Ti GPUs, 32 by 2,048 calibration tokens,
native MXFP4, exact selected-row BF16 `--lazy-ple`, and a 68 GiB host-memory
cgroup.

At the live receipt:

- layer 0 completed in 72.9 seconds;
- layer 1, the real PLE layer, completed in 275.2 seconds;
- layer 2 completed in 74.6 seconds;
- layer 3 was running;
- each completed layer artifact was 1,377,706,264 bytes.

Layer 1 completion is the critical boundary: the real PLE forward worked without
materializing the fused 95.37 GiB table. Two CUDA allocator retry warnings
occurred, but both affected layers completed and the process stayed alive.

Live receipt:
`build/reports/gptq-real-stage-a-live/receipt.json`.

Stage A remains incomplete; no Stage B artifact, real-model quality result, or
MRoPE parity is claimed.

## Local PLE storage-tier specification

Created
`docs/architecture/local-ple-storage-tiers.md` to make the deployment order
explicit:

1. Use exact BF16 PLE source shards directly from the internal SSD.
2. Keep the resident backbone native MXFP4.
3. Keep embedding and the language-model head BF16 for the first artifact.
4. Move the same immutable BF16 PLE shards to Optane after it is mounted.
5. Compare identical real traces before any media or format promotion.

The internal Data volume had approximately 2.6 TiB free. The estimated first
local working set is 161.47 GiB:

- 63.73 GiB MXFP4 backbone;
- 2.37 GiB BF16 embedding and LM head;
- 95.37 GiB exact BF16 PLE shards.

The specialized architecture is SHA-256
`ddf35b64ee8d2c5a61b9f8a40c31e6517acbf1e9f26e7e0b3959346eccc27807`.

A 128-entry verified source manifest was written to
`build/reports/local-ple-storage/ple-bf16-source-manifest.json`, SHA-256
`6e134b4cb51881af76b69d5b364ccc12dc1f83787f1ad3b9513ba58dbbba8ce2`.
It maps PLE `shard_0` through `shard_127` to their source model files, sizes,
and pinned Hub SHA-256 values.

The official tokenizer JSON differs from the diagnostic conversion tokenizer,
but both tokenizations of the complete 2,399,705-token corpus are identical.
The common token-stream SHA-256 is
`4efda5d4b5d75082d94cc2670265a0f4891cc566b49a04f275edc790c3a9b3a9`.
Therefore the current 32 by 2048 Stage A calibration windows are valid.

Receipt:
`build/reports/local-ple-storage/receipt.json`.

At the latest babysit check, real Stage A had completed 22 of 48 layers and
written 29 GiB of resumable layer artifacts. The process and `qwen-stage-a`
tmux controller remained alive.
