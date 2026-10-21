# EAGLE-4 BF16 and owned MXFP4 serving comparison

## Claim

The correctness-first EAGLE-4 chain backend now runs in the DFlash MLX
engine with absolute target positions, exact native greedy IDs on the pinned
short prompt, and no discarded top-k work on the greedy path. On this M4 Max,
the backend raises owned calibrated MXFP4 generation from 27.741 tok/s native
to 33.568 tok/s median at block two, but the incumbent DFlash2 selector at
effective block five remains the best measured local serving stack at
51.028 tok/s.

## Provenance and implementation

- Integration commit: `1687f5f` (`feat: add EAGLE-4 chain backend`).
- Canonical merge: `28400ea` (`merge: bring up EAGLE-4 chain backend`).
- Official EAGLE-4 source: `f6149b060b21f0912282d4410a01f2335f913372`.
- Draft artifact: `/Users/kearm/models/EAGLE-Corp-Qwen3.8-27B-EAGLE4`.
- BF16 target: `/Users/kearm/models/Qwen-Qwen3.8-27B-1d4bf0f`.
- Owned calibrated MXFP4 target:
  `/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4`.

The backend directly pairs the engine's staged/bonus token with the final
H-wide projected target feature. Each selected EAGLE token is then paired with
the prior normalized EAGLE hidden state. Rank-128 retrieval selects candidate
token IDs, and exact rows from the full draft head rescore those IDs. Greedy
selection, suppression, and capture top-k all map candidate positions back to
global token IDs.

The parent semantic audit found that the first v3 draft used zero-based
positions although this checkpoint uses RoPE. Official `DraftExtend` uses
absolute positions `seq_len + j`. The backend now derives the root position
from `TargetHiddenChunks.total_len` or a materialized `[1, sequence, H]`
context and increments it during each chain recomputation. Regression tests
cover both context forms.

An independent read-only review returned "approve with repairs" for the
root-feature pairing, candidate mapping, suppression, state isolation, and
runtime gating. Its requested sparse `TargetHiddenChunks` regression and
noted greedy top-k cleanup are both implemented. Its full ignored report is
`build/reports/eagle4/workers/backend-review-v3.txt`; the parent position fix
was applied in addition to that report.

## Validation

At the integrated worktree before merge:

```bash
PYTHONPATH=$PWD/runtime/dflash-mlx:$PWD \
uv run --no-project \
--with mlx==0.32.1 --with mlx-lm==0.31.3 --with pytest \
pytest runtime/dflash-mlx/tests/test_eagle4_backend.py \
runtime/dflash-mlx/tests/test_eagle4_draft.py \
runtime/dflash-mlx/tests/test_runtime_bundle.py -q
```

Result: 38 passed.

```bash
PYTHONPATH=$PWD/runtime/dflash-mlx:$PWD \
uv run --no-project \
--with mlx==0.32.1 --with mlx-lm==0.31.3 --with pytest \
pytest runtime/dflash-mlx/tests -q
```

Result: 1274 passed, 17 skipped.

Additional checks passed: `ruff format`, `ruff check`, `python3 -m
py_compile`, and `git diff --check`. At canonical merge head `28400ea`,
`just quality` passed all root, runtime, campaign, and CUDA-probe checks. The
generated receipt is ignored runtime evidence at
`build/reports/quality/receipt.json`; its scope explicitly does not prove
CUDA or Metal kernel performance or DFlash2 model quality.

## Short-prompt measurements

Prompt:

```text
Write a short Python function that adds two integers. Reply with code only.
```

Generation length was 32 tokens and the prompt token count was 67. Model-load
time is excluded by the comparison harness. The corrected absolute-position
runs live under `build/reports/eagle4/matrix-v3-position/`; only this
directory is authoritative for EAGLE-4 v3 performance. The older
`matrix-v3/eagle4-*` receipts predate the position repair.

| Target | Method | Block | Runs | Median tok/s | Acceptance | Exact native IDs |
|---|---|---:|---:|---:|---:|---|
| BF16 | EAGLE-4 v3 | b2 | 2 | 8.618 | 0.40625 | yes |
| MXFP4 | EAGLE-4 v3 | b2 | 3 | 33.568 | 0.40625 | yes |

The optimized MXFP4 run reached 33.899 tok/s. The best BF16 run reached
8.825 tok/s. Both optimized runs retained exact native greedy IDs and
acceptance 0.40625.

Reference measurements that do not depend on the EAGLE adapter remain:

| Target | Method | Effective block | Median tok/s | Acceptance | Exact native IDs |
|---|---|---:|---:|---:|---|
| BF16 | native | default | 8.957 | n/a | n/a |
| BF16 | DFlash2 | b5 | 22.468 | 0.750 | yes |
| MXFP4 | native | default | 27.741 | n/a | n/a |
| MXFP4 | DFlash2 | b5 | 51.028 | 0.750 | yes |

Both targets imply about 471 GB/s effective target-weight bandwidth in the
weights-only roofline. EAGLE-4 b2 reaches its 32.801 tok/s MXFP4 roofline but
remains below DFlash2 because this correctness-first backend recomputes the
short chain prefix each output, keeps no committed draft KV history, and does
not implement the official draft tree.

## Current serving decision

For local single-request serving, use the owned calibrated MXFP4 target with
the existing DFlash2 selector and requested block eight, which resolves to
effective block five. Its measured median is 51.028 tok/s and maximum is
53.254 tok/s on this prompt. EAGLE-4 is now a valid correctness baseline and
the target for the next optimization phase, not the current production
winner.

## Non-claims

- Exact native greedy IDs on one 67-token prompt do not prove broad output
  quality, perplexity, KL agreement, or multimodal behavior.
- These are short-context decode measurements; they do not establish
  long-context prefill or decode performance.
- The EAGLE-4 adapter intentionally lacks committed draft KV history, full
  tree attention, DDTree proposals, sampled proposals, and CopySpec support.
- The native and DFlash2 reference medians are repeated from the clean
  `matrix-v3` set because those paths do not use the repaired EAGLE backend.
- No CUDA throughput result transfers to M4 Max.
- No external EAGLE-4, MLX, or Hugging Face repository was modified.
