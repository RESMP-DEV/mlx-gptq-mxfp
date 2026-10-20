# DFlash2 QAT prescribed-grid export

Date: 2026-10-20
Agent: Codex

## Finding

Requantizing the effective QAT weights is not idempotent at every E8M0 scale
boundary. A group whose original maximum is 2.2 selects scale `0.5`; QAT maps
that endpoint to `2.0`. Re-running scale selection on `2.0` selects `0.25`.
The old exporter therefore changed the prescribed grid even though every stored
value was already an MXFP4 value.

The trainer now emits `qat_mxfp4.safetensors` beside `model.safetensors`.
It stores the exact uint8 E2M1 codes and uint8 E8M0 scale bytes chosen for the
QAT-effective selector projection and every QAT LoRA target. The exporter's
`--qat-source` path consumes those tensors, validates layout and dtype, and
fails if metadata is absent. The non-QAT path remains ordinary RTN.
The exporter also dequantizes every prescribed pair and requires exact equality
with the checkpoint tensor. A blended checkpoint therefore fails rather than
packing the unblended run's stale codes and scales.

## Boundary check

The deterministic scale-boundary check used one `8x64` tensor with a group
maximum of 2.2:

```text
prescribed_first_scale = 0.5
naive_first_scale = 0.25
prescribed_max_abs_error = 0.0
naive_scale_mismatch_fraction = 0.3125
```

The exported prescribed codes dequantize exactly to the QAT-effective tensor;
the old selection rule would disagree on 31.25% of scales in this case.

## Validation

```text
uvx ruff check \
  mlx_gptq/dflash2_distill.py \
  tools/export_rtn_mxfp4.py \
  tests/test_dflash2_distill.py \
  tests/test_export_rtn_mxfp4.py
exit 0

.venv/bin/python -m pytest tests/ -q
66 passed, 5 skipped
```

The later count includes the run validator; the focused exporter test has four
passing cases, including rejection of a stale grid after blending.

## Non-claims

- Existing old artifacts are not silently repaired; re-export them from a
  source run that emits prescribed-grid metadata.
- This proves export-grid identity, not live acceptance.
- No rejected or newly exported candidate is promoted or uploaded.
