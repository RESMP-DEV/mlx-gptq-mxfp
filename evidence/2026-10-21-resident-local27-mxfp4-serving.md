# Resident local27 MXFP4 serving and scheduler analysis

## Claim

The owned calibrated Qwen3.8-27B MXFP4 target and the campaign-owned
GPTQ-MXFP4 DFlash2 drafter are now resident in a loopback LaunchAgent service.
The service uses only our own quantizations, keeps 19.4 GiB of MLX state
active, restores after ordinary crashes, and supports explicit reversible
eviction. A first real request generated 18 tokens at 54.19 decode tok/s with
0.778 acceptance and 4.5 tokens per verify cycle.

## Provenance

- Target: `/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4`.
- Drafter:
  `/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-1981cycles`.
- The drafter is already a campaign-owned native MXFP4 artifact. The server's
  `draft_quant: none` line means no runtime quantization is applied, not that
  a community quant is in use.
- Service label: `dev.resmp.local27-mxfp4`.
- Admission label: `dev.resmp.local27-admission`.
- Endpoint: `http://127.0.0.1:8081`.
- Bounded queue endpoint: `http://127.0.0.1:8080`.
- Wrapper: `scripts/local27_mxfp4_service.sh`.
- LaunchAgent plist: `scripts/dev.resmp.local27-mxfp4.plist`.
- Admission plist: `scripts/dev.resmp.local27-admission.plist`.
- Admission proxy: `scripts/local27_admission_proxy.py`.
- Durable analysis:
  `build/reports/local-mxfp4-serving/scheduler-design.md`.

## Before this change

The claimed resident workload was not actually resident. A listener and
open-file audit found only LM Studio's Nomic embedding model on port 1234;
no process held the 27B model files. The earlier 27B MLX and CCR servers were
temporary capacity probes and had exited.

## Live verification

`launchctl print gui/501/dev.resmp.local27-mxfp4` reports the service running.
The first launch attempt failed safely because launchd's sanitized PATH could
not find `uv`; the wrapper now uses the absolute binary
`/Users/kearm/.local/bin/uv` and fails closed if it is missing.

`GET /metrics` reported:

- Model: `/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4`.
- Draft:
  `/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-1981cycles`.
- MLX active memory: 19.425 GiB.
- Physical footprint: 19.960 GiB.
- RSS: 19.793 GiB.
- Wired limit: 115.449 GiB.

A real greedy chat request returned a valid Python function, 27 prompt tokens,
18 completion tokens, and `finish_reason=stop`. Server metrics for that
request:

- Decode: 54.188 tok/s.
- Acceptance: 0.777778.
- Tokens per verify cycle: 4.5.
- TTFT: 0.589 s.
- Full request wall time: 0.921 s server-side, 0.970 s including curl.
- Prefix cache: one insertion, zero evictions.

## Residency policy

The model should remain resident by default. Other agents may evict it only
for substantially larger work by running:

```bash
/Users/kearm/mlx-stack/scripts/local27_mxfp4_service.sh evict
```

They must restore it afterward:

```bash
/Users/kearm/mlx-stack/scripts/local27_mxfp4_service.sh restore
```

Ordinary crashes are handled by launchd KeepAlive. Explicit eviction uses
`launchctl bootout`, so launchd does not immediately fight the large job.

## Eviction, restore, and concurrency

Eviction stopped both launch agents, closed ports 8080 and 8081, and reclaimed
the model worker memory. Restore reloaded both services. The corrected
`/readyz` endpoint passed only after the model backend answered, and the worker
returned to 19.76 GiB RSS with the correct target and drafter.

Direct bursts above roughly four to eight clients could receive connection
resets in the MLX HTTP handler. Phase-one admission now queues non-streaming
requests and bounds backend concurrency at four. In a 126-request matrix at
concurrency 1, 2, 4, 8, 16, and 32, every request completed with
`finish_reason=stop`; the proxy reported zero backend errors.

Median wall-clock aggregate completion rates through the serial DFlash
generation worker were 17.13, 19.09, 19.42, 17.05, 15.30, and 11.16 tok/s at
concurrency 1 through 32. The proxy fixes resets and admission, not DFlash
batch throughput. The earlier 142 aggregate tok/s result came from plain
batched MLX LM; choosing single-stream DFlash versus batched plain MLX is the
next serving-policy decision.

## Scheduler analysis

The analysis maps SGLang's zero-overhead batch scheduler, DeepSeek's
prefill/decode and communication-overlap design, and Strata's batch slots,
chunk-boundary yielding, decode-share, and conversation parking onto MLX.

The mechanisms most applicable to this one-node service are:

1. Two admission classes: short decode-heavy requests versus 8k-plus prefill.
2. Chunk-boundary yielding so a long prompt degrades short requests instead
   of stalling them.
3. A decode budget between long-prefill chunks, analogous to Strata's
   `STRATA_BATCH_DECODE_SHARE`.
4. Prefix-cache affinity for repeated agentic turns; L1 and L2 caches are
   already enabled.

Expert parallelism, multi-GPU pipeline groups, and cross-node load balancing
do not transfer to one M4 node. SGLang's exact zero-overhead result is a
CUDA-graph result and does not numerically transfer.

An independent Strata/DeepSeek audit found the same boundary: MLX-LM marks a
provider with a loaded drafter non-batchable, so the DFlash server serializes
requests. Strata's directly portable policies are short-first admission with
aging, cooperative yield at chunk boundaries, prefix-aware slot placement, and
conversation parking. DeepInfer-style zero-overhead scheduling should wait
until a batched engine exists; its CUDA metadata remap is not portable as-is.

## Batched-control result

A temporary second process running plain batched MLX LM was also measured. It
was stopped afterward; the resident DFlash service remains loaded.

- At four simultaneous 128-token requests through a four-way gate: 38.89
  wall-clock aggregate tok/s.
- At 16 requests: 36.97 tok/s.
- At 32 requests: 23.16 tok/s.
- A separately preconnected 12-way, 128-token run reached 35.80 tok/s.
- Evicting the idle DFlash process before a repeat 12-way run did not improve
  it (31.69 tok/s), so the difference from the earlier 142 tok/s receipt is
  not simply memory contention.

This control did not reproduce the earlier 142 tok/s operating point. It used
a different request profile and an admission-gated server, so it is not a
contradiction; it means the next batched-lane experiment must reproduce that
receipt's exact workload and direct CCR arrival pattern before selecting a
concurrency engine.

## Non-claims

- The 54.19 tok/s result is single-request decode throughput; wall-clock
  one-request completion throughput is lower.
- Phase-one bounded admission is implemented, but preemption classes,
  chunk-boundary yield, decode-share, and streaming admission are not.
- No long-context yield experiment has run.
- Strata and DeepSeek throughput numbers do not transfer to M4.
- The service is loopback-only and is not exposed to external networks.
