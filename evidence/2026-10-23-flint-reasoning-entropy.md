# Flint top-logprob entropy controller

Date: 2026-10-23
Agent: Codex

## Scope

Port the measured omni-use reasoning-shortening mechanism to the owned Flint
MLX serving surface without changing default generation behavior. The mechanism
is opt-in and benchmark-driven; the server still returns ordinary completions
unless the controller is explicitly invoked.

## Source mechanism

The source implementation is `RESMP-DEV/omni-use` commit
`c2ff674` in:

- `scripts/benchmark_reasoning_entropy.py`
- `scripts/tests/test_benchmark_reasoning_entropy.py`
- `evidence/2026-10-23-holo4-logprob-entropy-reasoning-controller.md`

It combines:

1. *Think Just Enough* sequence entropy: normalize each token's top-20
   logprobs and compute Shannon entropy in bits.
2. *Dynasor/Certaindex* real-prefix probing: cut only after real generated
   reasoning, then force a final-answer probe.

The Holo4 reference result used 32 calibration and 32 held-out ScreenSpot
desktop tasks. It retained 32/32 hits, reduced mean tokens from 175.781 to
128.125 (27.11%), and reduced median wall time from 2.186 s to 1.319 s. Its
receipt SHA-256 is
`8ab5e174b355f59af2d01d9f0962c248c03f40db2d647e580b5ba43b96d4307d`.

That result is source evidence only; it does not transfer automatically to
Flint.

## MLX-LM contract

The owned MLX-LM server previously capped `top_logprobs` at 11, too small for
the measured top-20 statistic. Commit
`dabc78019ec30cacac36a787891010a089912e8e` adds `MAX_TOP_LOGPROBS = 20`.

Validation:

```text
tests/test_server.py: 41 passed
ruff check: passed
ruff format --check: passed
```

## Flint implementation

Added:

- `mlx_gptq/reasoning_entropy.py`
- `scripts/flint_reasoning_entropy.py`
- `scripts/tests/test_flint_reasoning_entropy.py`

The controller:

- loads JSONL tasks with `prompt`, `expected`, optional `id`, and optional
  `probe_suffix`;
- generates a real checkpoint prefix with top-20 logprobs;
- computes normalized mean Shannon entropy;
- calibrates `mean_correct` or `zero_observed_error` thresholds;
- probes after the real prefix;
- cuts only when entropy exists and is at or below threshold;
- always continues when entropy is missing;
- reports baseline and policy token count, wall time, hit rate, cuts,
  continues, token delta, and accuracy delta.

The cut request reuses the real generated prefix. It never synthesizes
reasoning text.

Unit validation:

```text
scripts/tests/test_flint_reasoning_entropy.py: 6 passed
ruff check: passed
ruff format --check: passed
```

## Live Flint smoke

The live run used the repaired public 25-shard Flint MXFP4 artifact, the local
MLX-LM server, raw completion prompts, two calibration tasks, two held-out
tasks, a 16-token checkpoint, a 16-token probe, and a 64-token baseline cap.

Calibration:

- correct probes: 2/2
- threshold: 0.584941672754121 bits

Held-out policy:

- cuts: 2
- continues: 0
- mean tokens: 64.0 baseline, 32.0 policy
- mean wall time: 0.7070 s baseline, 0.3645 s policy
- measured hit count remained 1/2

This is a small wire-path smoke, not a quality or general speed result. One
miss used the expected string `4` while the model wrote `four`; the task
fixture tests transport, not normalization quality.

Artifacts:

- `build/reports/flint-entropy/simple-raw-2x2-c16-p16-max64-final.json`
- `build/reports/flint-entropy/simple-raw-2x2-c16-p16-max64-final.log`

The JSON SHA-256 is
`e4226c664eed238b4f5d6660dd67d88e6707d86c7970daa4dc4e82c9364977ca`.

## Boundary finding

Raw `/v1/completions` prompts worked end to end. A probe fixture containing
Flint role markers caused the current server's segment planner to report
`Sequence N has an empty prompt` on a continuation and terminate the generation
thread. This is a separate prompt-segmentation defect. The controller works
with raw prompts, but role-marker continuation should be repaired before the
gate is used with templated chat prompts.

## Non-claims

- The two-task smoke does not establish task quality or production savings.
- Threshold and savings must be recalibrated for each Flint workload, prompt
  format, checkpoint, and quantization.
- Entropy is a confidence signal, not proof of correctness.
- The controller does not change the published model weights.

## Exact-cache continuation repair and chat-marker replay

The role-marker blocker is now repaired in `RESMP-DEV/mlx-lm` commit
`de15a0e` with evidence in `evidence/2026-10-23-exact-prompt-cache-continuation.md`.
An exact prompt-cache hit previously consumed every token segment and caused
`Sequence N has an empty prompt`; the server now re-prefills the full prompt and
keeps the generation thread alive. The dedicated HTTP regression fails before
the fix and passes afterward; the full server suite passes 42 tests.

The same four-task chat fixture was replayed against the fixed server at
`http://127.0.0.1:8083` with the 21 GB `/Users/kearm/mlx-stack/build/flint-resharded`
checkpoint, two calibration tasks, two held-out tasks, a 16-token checkpoint,
a 16-token probe, and a 64-token cap.

Result:

- all calibration and held-out requests completed; the generation thread
  stayed alive;
- calibration threshold: `0.15389435978063082` bits;
- held-out cuts: 0;
- held-out continues: 2;
- mean tokens: 64.0 baseline, 64.0 policy;
- mean wall time: 0.8057 s baseline, 0.7856 s policy;
- measured hit count: 1/2 in both arms.

This replay proves the formerly fatal chat-marker path now works end to end.
It does not show a token or latency win: the calibrated threshold was below
both observed checkpoint entropies (`0.30838` and `0.22176`), so the controller
correctly chose continue twice. A representative agentic coding calibration set
is still required before claiming savings.

Artifacts:

- `build/reports/flint-entropy/simple-chat-fixed-2x2-c16-p16-max64.json`
- `build/reports/flint-entropy/simple-chat-fixed-2x2-c16-p16-max64.log`
