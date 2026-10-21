# FreeToken serving and batched-MTP architecture receipt

## Scope

This receipt records the bounded analysis of
`FlashML-org/FreeToken` at `cc4aa085b6eaeda01c0441fb3ecb9cc843b3e465`
and turns the accepted findings into the first backend-neutral serving
contracts in this repository. It also maps the pieces that can be reused by
MLX and Strata-like engines. No external repository was modified.

Operators and split:

- Parent integration: Codex.
- FreeToken mechanism audit: CCR worker, read-only.
- macOS verification matrix audit: CCR worker, read-only.
- MLX batched-MTP seam map: CCR worker, read-only.
- Strata portability map: CCR worker, read-only.

The worker reports are retained under
`build/reports/freetoken/workers/`; their claims below were checked against
the named source paths.

## Decisive findings

### FreeToken has no serving-loop MTP

FreeToken's useful transfer is scheduling and cache architecture, not
speculation. At the pinned revision:

- `python/freetoken/models/qwen4_exp/weight.py:9` explicitly drops `mtp.*`
  as the speculative head.
- `python/freetoken/models/qwen3_5_moe/weight.py:51` drops `mtp.*` keys.
- `python/freetoken/models/glm4_moe/config.py:28` drops the trailing MTP
  layer.
- `python/freetoken/models/deepseek_v41/__init__.py:7` states that DSpark
  speculative decoding is not served.
- A search of `freetoken/scheduler`, `freetoken/server`,
  `freetoken/daemon`, and `freetoken/engine` found no drafter, verify
  window, acceptance path, or MTP scheduler.

Therefore the direction is inverted from the initial premise: Strata already
has portable batched-MTP policy machinery, while FreeToken contributes
admission, cache capability, and memory-budget discipline.

### FreeToken mechanisms worth porting

Portable policy:

- `scheduler/prefill.py:43-109`: multi-currency admission with a token
  budget, reserved decode capacity, page-aligned KV accounting, hybrid GDN
  state slots, and SWA-pool reservation.
- `scheduler/cache.py:32-79`: one cache manager with explicit capability
  hooks, including runtime rebuild, prefill budget, alignment, and replay
  tokens.
- `kvcache/linear_state_pool.py:34-79`: recurrent conv and SSM state
  indexed by a request slot, rather than a per-request Python cache object.
- `kvcache/hybrid_radix_cache.py`: separate full-KV and linear-state
  refcount and eviction currencies.
- Abort handling marks a request and releases resources only after the
  in-flight forward completes. This is the correct cancellation boundary for
  a batched speculative engine.

CUDA-specific and not portable to MLX:

- CUDA streams/events, CUDA graph padding, device-side LRU admission,
  pinned-memory batch copies, Triton and native kernels, expert banks, and
  PCIe/CPU co-execution policies.

### Strata mechanisms worth absorbing

The highest-value artifact is `include/strata/spec/confidence_varlen.hpp`
and `src/spec/confidence_varlen.cpp`. It is a pure host contract:

- every slot contributes one golden row;
- draft proposals are ranked by cumulative conditional confidence;
- a global row budget is respected;
- output is contiguous slot-major row counts and offsets;
- the planner neither drafts, verifies, nor touches runtime state.

The existing `src/spec/confidence_varlen_test.cpp` passed unchanged on this
machine. Strata additionally supplies useful shapes for row validation,
cooperative prefill yield, conversation checkpoints, draft policy, and
verifier commit prefixes, but its concrete drafter and verifier are
CUDA-bound.

### Current MLX boundary

Against the pinned MLX-LM 0.31.3 contract:

- `mlx_lm/server.py:370-381` sets `is_batchable = draft_model is None` and
  also requires every default cache entry to expose `merge`.
- `mlx_lm/server.py:685-686` makes seeded requests non-batchable.
- `mlx_lm/server.py:813-815` sends non-batchable requests to
  `_serve_single`.
- DFlash's `DFlashResponseGenerator` overrides `_serve_single`, and its
  provider never opts into MLX-LM's batch generator.
- The plain Qwen3.5 target cache is batchable: linear layers use
  `ArraysCache`, which supports merge/filter, and full-attention layers merge
  through `BatchKVCache`.
- The speculative target cache is not batchable as-is:
  `RecurrentRollbackCache` has `extract` and scalar `rollback`, but no
  `merge` or `filter`.
- `engine/ddtree.py` already contains the local precedent for cloning a
  cache into rows and issuing a stacked verify. That work is tree-row and
  oracle-path only; DFlash2 disables DDTree.

The architectural consequence is a two-lane design:

1. A target-only batch lane can use the model's normal mergeable cache.
2. A speculative lane needs a native batch session and per-row rollback;
   flipping the provider flag alone would be incorrect.

## Implemented contracts

`runtime/dflash-mlx/dflash_mlx/server/serving_contracts.py` adds a
backend-neutral module with no MLX import:

- `plan_confidence_varlen`: the Strata-compatible confidence-ranked ragged
  row planner.
- `accepted_draft_lengths`: greedy acceptance and committed-row calculation.
- `AcceptancePlan.rollback_groups`: groups rows by committed length so a
  backend can replay or trim equal-acceptance groups without a new kernel.
- `plan_short_first_admission`: short-first admission with bounded aging and
  cancellation separation.
- compact JSON serialization for the confidence plan, so Strata or another
  engine can consume the same metadata.

This is implemented and host-tested, but it is not yet wired into live model
execution. It is the scheduling contract that the later batch executor will
consume.

## Staged implementation

### Stage 0: baseline and exact replay

Replay the earlier 142 tok/s plain-MLX workload exactly from
`evidence/2026-10-20-local-27b-ccr-fallback-capacity.md`. Do not choose a
lane or scheduler on the later non-reproducing control alone.

### Stage 1: single-process target batch lane

Keep one resident target-model instance and route target-only requests
through a batch lane using normal `ArraysCache`/`KVCache` semantics.
Speculative requests remain on the serial DFlash lane. The lane split is
per request, not a provider-global boolean.

Required seams:

- `dflash_mlx/server/batch_lane.py`
- `dflash_mlx/server/scheduler.py`
- `DFlashResponseGenerator._generate`
- per-request response queues and accounting

### Stage 2: cross-request verify spike

Before touching live serving, batch independent speculative rows off-path:

- clone or stack per-row target state;
- issue one stacked verify forward;
- calculate per-row acceptance;
- group rows by committed length;
- replay GDN tape and trim KV by group;
- compare each row to a serial run.

Start with equal-length or shared-prefix histories. Arbitrary histories
require per-row offsets, paged KV, rotating KV, or a measured shared-prefix
attention form. Do not claim general ragged-history batching until that gate
is measured.

The GDN rollback decision gate is narrower and should run first:
`tape_replay_kernel` already accepts a per-step mask, so heterogeneous
acceptance may be replayable in one call. That path has not been exercised
with per-row lengths, and conv-state reconstruction is still scalar today.
The implemented `rollback_groups()` contract is the safe grouped fallback;
a single masked replay is the optimization to prove or reject. Arithmetic
from the target config puts live GDN recurrent state near 144 MiB per row,
before tape and full-attention KV, so batch capacity must be measured rather
than inferred.

### Stage 3: batched MTP lane

Use the owned MTP artifacts only:

- BF16: `/Users/kearm/models/Qwen3.8-27B-MLX-MTP-BF16-1d4bf0f`
- local W4: `/Users/kearm/models/RESMP-DEV-Qwen3.8-27B-MTP-W4-local`

The MTP head consumes target final hidden state and target embedding, has
one layer, and shares the target LM head. Its head pass is
bandwidth-dominated, which favors batching, but its current measured single
request speed remains below incumbent DFlash2. It is a candidate lane, not a
replacement.

### Stage 4: production scheduler and cache affinity

Promote the pure contracts into the live scheduler:

- multi-currency prefill admission from FreeToken;
- decode reservation and chunk-boundary yield;
- cancellation after forward completion;
- prefix-aware slot placement;
- capability-declared caches;
- root-pinned conversation cache with LRU leaves;
- separate live and snapshot linear-state slots.

## Correctness and performance gates

Any live batched speculative lane must pass all of:

1. Exact greedy token-ID equality against the serial target lane.
2. Row independence: a prompt must produce the same IDs alone and with
   adversarial co-tenants.
3. Rollback equality: post-cycle GDN and KV state equals the equivalent
   serial state for forced mixed acceptance lengths.
4. Prefix-cache equality: cold and hydrated-prefix runs produce identical
   IDs, with no cross-row snapshot contamination.
5. Cancellation safety: one disconnected row cannot corrupt or free state
   under an in-flight forward.
6. Paired concurrency matrix at 1/2/4/8/16/32 with same-run serial and batch
   controls, reporting aggregate tok/s, TTFT, p50/p95 latency, acceptance,
   and peak memory.
7. No single-stream regression beyond a stated bound before promoting a
   batch lane.

These are future gates for live execution; they have not been run in this
implementation slice.

## Bounded verification

Commands and results:

```text
cd /Users/kearm/mlx-stack/.worktrees/freetoken-serving/runtime/dflash-mlx
PYTHONPATH=. uv run --no-project \
  --with pytest --with mlx==0.32.1 --with mlx-lm==0.31.3 \
  pytest tests/test_serving_contracts.py -q
10 passed
```

The test includes 100 seeded brute-force comparisons of the confidence
ranking in addition to the Strata test cases.

```text
cd /Users/kearm/Strata
./build/confidence-varlen-main/confidence_varlen_test
PASS
```

This ran the pre-existing host test without modifying the dirty Strata tree.

```text
cd /Users/kearm/research/ml/FreeToken
PYTHONPATH=python:. uv run --no-project \
  --with pytest --with fastapi --with httpx --with pydantic \
  pytest tests/daemon -q -p no:cacheprovider --noconftest
37 passed, 1 warning
```

Without `--noconftest`, the same daemon suite had 37 passing test bodies and
37 teardown errors because the global autouse fixture imports
`freetoken.layers.quantization`, which imports Torch. This corrects the
worker's initial claim that the daemon tier needs no Torch.

`flashlib==0.4.0` is installable on macOS, but importing its required
`flashlib.kernels.slot_cache` fails because `flashlib.kernels.__init__`
eagerly imports unrelated Triton modules and macOS has no Triton wheel.
Higher FreeToken scheduler suites therefore remain source-audited, not
runtime-verified, on this M4 Max host.

Additional local verification is recorded by the quality receipt after this
branch is checked.

```text
cd /Users/kearm/mlx-stack/.worktrees/freetoken-serving
just quality
passed=true recorded_at=2026-10-21T18:00:14.783512+00:00
```

The receipt at `build/reports/quality/receipt.json` covers root lint and
tests, the complete DFlash runtime tests, campaign tests, and the CUDA probe
compile.

## Non-claims

- No FreeToken or Strata file was modified.
- No upstream pull request, issue, or push was made.
- No FreeToken inference or CUDA kernel was executed.
- No MTP model artifact was loaded in this slice.
- No new throughput or latency claim is made for the implemented contracts.
- The confidence planner is scheduling logic only; it does not prove batched
  target execution or rollback correctness on Metal.
- The resident local27 service remained loaded and was not evicted.
