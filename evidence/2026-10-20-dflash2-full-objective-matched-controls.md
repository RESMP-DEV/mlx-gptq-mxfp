# DFlash2 full-objective matched controls

Date: 2026-10-20
Agent: Codex
Base revision: `c7671d1`

## Change

The corrected trainer now accepts `--validation-capture-dir`. It rejects a
capture whose prompt indices overlap training, puts both datasets into the same
ContextOnly cache order, evaluates only the external validation capture, and
hashes both capture indexes into the resume fingerprint. This makes QAT and
BF16/RTN arms share one immutable session-grouped validation set.

`tools/export_dflash2_capture_split.py` exports prompt-disjoint training and
validation captures. Cycle limits take contiguous prefixes within each prompt,
not random samples. Random cycle sampling can reset a ContextOnly cache at a
later logical position and silently change RoPE and sliding-window behavior;
prefixes preserve the replay contract. The split receipt records both the full
prompt partition and the prompts actually selected.

## Validation

Commands and results:

```text
uvx ruff check \
  mlx_gptq/dflash2_distill.py \
  tools/export_dflash2_capture_split.py \
  tests/test_dflash2_distill.py \
  tests/test_export_dflash2_capture_split.py
exit 0

.venv/bin/python -m pytest tests/ -q
58 passed, 5 skipped
```

A real capture smoke used the 896-prompt, 62,026-cycle full-logit capture with
seed 43 and validation fraction 0.2. It exported eight training cycles and two
validation cycles from disjoint prompt sets. Every selected group remained
contiguous in dataset order, and the first target-logit tensor had shape
`[5, 248320]`, proving full-vocabulary targets. The ignored output is:

```text
/Users/kearm/mlx/build/reports/dflash/capture-prefix-split-smoke-20261019
```

## Matched QAT experiment contract

The first post-smoke arms must use the same base model, exported training and
validation captures, projection initialization, LoRA geometry, optimizer,
seed, objective weights, replay setting, and live prompts. The BF16 arm trains
without `--qat-mxfp4` and is RTN-projected after merge; the QAT arm uses
`--qat-mxfp4` during training and export. Offline validation selects candidates
only. Promotion still requires exact target parity and positive paired live
acceptance versus the incumbent, with more wins than losses and the established
two-sided paired-test threshold.

## Non-claims

- This is training-control plumbing, not a model-quality result.
- No QAT or BF16/RTN artifact was promoted or uploaded.
- The small exported split proves layout and full-logit presence, not objective
  quality or runtime acceptance.
- The active 128-cycle CPU smoke was loaded before this change and was not
  restarted; its eventual receipt still evaluates the prior corrected trainer.

## Update: bounded complete session groups

Added after the first smoke exposed the severed `fc` gradient: the exporter also
accepts `--train-prompt-limit` and `--validation-prompt-limit`. This selects a
deterministic subset of the already shuffled prompt partition before applying
contiguous cycle-prefix limits. A smoke can therefore guarantee several cycles
per prompt instead of spreading a tiny cycle budget across hundreds of
first-cycle entries.

Validation:

```text
uvx ruff check tools/export_dflash2_capture_split.py \
  tests/test_export_dflash2_capture_split.py
exit 0

.venv/bin/python -m pytest tests/test_export_dflash2_capture_split.py -q
3 passed
```

The fresh smoke split at
`/Users/kearm/mlx/build/data/dflash2-prefix-split-p16-p4-20261020` has 16
training prompts/128 contiguous cycles and four validation prompts/16 cycles.
Its per-prompt dataset indices are contiguous, with 3-12 training and 2-5
validation cycles per group. This receipt still does not claim data quality or
runtime acceptance.
