# Target batch hardening and recent-model preparation receipt

## Scope

This slice turns the FreeToken/Strata serving analysis into three working
pieces:

1. A default-off target-only continuous-batch process mode.
2. A heterogeneous per-row GDN rollback primitive.
3. A repeatable target-batch probe and recent-model integration lane.

The resident DFlash service on ports 8080/8081 was not evicted or restarted.
An isolated validation service used port 8082 and was stopped after the
measurement.

## Target-only batch mode

Implemented `--target-batch-only`:

- Loads only the owned target model.
- Deliberately skips DFlash speculative hooks.
- Verifies every default target cache exposes `merge`.
- Sets MLX-LM `is_batchable` only when the target-only cache contract holds.
- Disables process-local DFlash prefix cache/L2 state.
- Reports `mode: target_batch`.
- Adds explicit decode/prompt concurrency flags.
- Defaults off and rejects an explicit `--draft-model`.

The hook boundary was found by a real failed run, not inferred. With
speculative full-attention hooks installed, MLX-LM `BatchKVCache` reached the
DFlash hook and failed because the hook converted a batched offset array to a
scalar. Target-only loading now passes
`install_speculative_hooks=False`; speculative DFlash loading still installs
the hooks.

Real isolated validation:

```text
model=/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4
port=8082
mode=target_batch
draft=null
```

A greedy code request completed normally with 18 tokens. The isolated service
was then stopped.

### 32-way measured result

Command shape:

```text
32 concurrent chat-completion requests
20 ms per-index arrival stagger
temperature=0
max_tokens=128
prompt="Write the integers from 1 to 100, separated by commas. Output only the list."
direct endpoint=http://127.0.0.1:8082
```

Result:

```text
requests=32
completion_tokens=4096
wall_s=29.47823362494819
aggregate_completion_tok_s=138.94998092875718
median_request_completion_tok_s=4.425550890098414
p50_request_wall_s=28.92295312450733
all_finish=true
finish_reason=length for every request
```

Metrics after the run:

```text
mlx_active_gb=21.48874555
mlx_cache_gb=8.587141557
mlx_peak_gb=30.388860582
phys_footprint_gb=31.681673664
rss_gb=18.849349632
```

This reproduces the prior direct MLX-LM operating point inside the owned
DFlash process, without a community quant and without evicting the resident
service. It is not a CCR-route result and does not claim streaming behavior.

### Serial/batch agreement probe

The resident DFlash lane on 8080 and isolated target-batch lane on 8082 each
generated 48 greedy tokens from the same prompt. Visible completion content
was exactly equal. The batch request reported 32 cached prompt tokens versus
zero on the serial request. This is exact visible-content agreement, not a
token-ID-level parity gate.

## Heterogeneous GDN rollback

Added `RecurrentRollbackCache.rollback_batch(sequence[int])`:

- validates one acceptance count per tape row;
- builds a per-row, per-step replay mask;
- invokes `tape_replay_kernel` once for all rows;
- reconstructs each row's convolution state at its own accepted boundary;
- clears transients like scalar rollback.

Synthetic Metal tests compare batched rollback against independent one-row
rollback for acceptance counts 0, 1, and 3. Both recurrent state and conv
state are exactly equal. The primitive is not yet wired into live batched
speculative serving.

## Durable probe

Added `scripts/local27_target_batch_probe.py` with:

- matrix and health-only modes;
- configurable concurrency, stagger, timeout, prompt, and output;
- per-request timing and errors;
- aggregate and median rates;
- all-finish and identical-content gates;
- before/after `/metrics` capture;
- JSON receipt output.

## Recent-model candidate scan

Live Hugging Face metadata/config/index inspection ranked compatibility, not
model quality or product fit:

1. `ornith-ai/Ornith-1.5-122B`
   - created 2026-10-16, updated 2026-10-21;
   - MIT tag, ungated;
   - `qwen3_5_moe`, the exact hybrid-GDN family used by this stack;
   - 125,086,497,008 BF16 parameters;
   - 48 layers, 256 fused experts, 8 active per token;
   - vocab 248,320, matching Qwen3.8-27B;
   - one MTP layer and 785 `mtp.*` tensors;
   - a separate 930M DFlash2 drafter exists at
     `incoai/Ornith-1.5-122B-DFlash2`.
2. `InternScience/Sage-A1.5`
   - Apache-2.0 and `qwen3_5_moe`, but individual expert tensors, low
     adoption, and the exact verify-linear-excluded 40/2048/16/2 shape.
3. `openbmb/MiniCPM-V-4.7-35B-A3B`
   - Apache-2.0 with a Qwen3.5-MoE text tower, but no `minicpmv4_7` MLX-LM
     module and no base-checkpoint MTP tensors despite config metadata.

`deepseek-ai/DeepSeek-V4.1-Flash-1017` is intentionally deferred. It is MIT,
very recent, and uses native FP4 experts with UE8M0 32-wide scales, but its
`engram_layer_ids` hard-fail in the current DeepSeek MLX model and its 510 GB
artifact make it a later architecture project rather than the first
integration.

The full live scan is retained at
`build/reports/freetoken/workers/recent-model-scout.txt`.

### Paused Ornith transfer

The compatibility scan led to a premature transfer of this source:

```text
ornith-ai/Ornith-1.5-122B
revision=7681ef608624e7532d8164128bcffdbdbaf1b1af
```

HFD transfers were started to both machines, then stopped after review:

- Main: `/Users/kearm/models/Ornith-1.5-122B-BF16-7681ef`
- B550: `/home/kearm/models/Ornith-1.5-122B-BF16-7681ef`

Both HFD processes initially stalled in IPv6 `SYN_SENT`. A temporary IPv4
resolver wrapper restored transfer. The launchd/nohup jobs were then removed.
At the stop snapshot the partial trees were about 34 GiB on main and 16 GiB
on B550. They are resumable but not selected, not complete, and not size
verified.

Ornith's only defensible research use is a compatibility stress test of a
newer Qwen3.5-MoE checkpoint with MTP and 256 experts. It does not solve a
new serving use case, has no measured quality advantage, and is not a
replacement for the resident Qwen3.8-27B service. Do not resume it without an
explicit model-selection decision.

No external or community quant is an acceptance baseline. The first-party
Ornith MLX/NVFP4 artifacts are reference candidates only; calibration must
come from this stack.

## Verification

Focused commands:

```text
runtime target-batch/config/metrics tests: 83 passed
heterogeneous rollback tests: 2 passed
target-batch probe tests: 4 passed
```

The full quality receipt is regenerated after this branch is finalized.

```text
cd /Users/kearm/mlx-stack/.worktrees/freetoken-serving
just quality
passed=true recorded_at=2026-10-21T18:46:43.082653+00:00
```

The quality receipt covers root lint/tests, the full DFlash runtime tests,
campaign tests, and CUDA probe compilation.

## Non-claims

- The 138.95 tok/s result is direct loopback, not through CCR.
- Streaming request behavior is not claimed.
- Exact token-ID parity was not captured; only exact visible greedy content.
- `rollback_batch` is proven synthetically and not yet used by live serving.
- The Ornith transfer is incomplete and unverified at receipt time.
- No Ornith inference, calibration, conversion, or upload has run.
- The resident local27 DFlash service remains the default loaded service.
