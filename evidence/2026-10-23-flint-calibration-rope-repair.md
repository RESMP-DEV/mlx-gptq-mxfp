# Flint calibration RoPE repair

Date: 2026-10-23
Agent: Codex

## Finding

The eager PyTorch calibration skeleton in `mlx_gptq/flint_torch.py` rotated
adjacent even/odd RoPE lanes. The MLX-LM Flint runtime uses
`nn.RoPE(traditional=False)`, the NeoX-style half-split layout. A read-only
audit confirmed the mismatch with a direct MLX comparison.

This matters for future layer-sequential GPTQ, RCO materialization, or GSQ
optimization: calibration and mask-search Hessians must be collected under the
same rotary layout as deployment.

## Change

`_rope` now splits each head dimension into first and second halves:

```text
x0 = x[..., :half]
x1 = x[..., half:]
```

It writes the rotated first half and second half back to the same halves.

A regression test creates random `[1, 2, 3, 16]` values and compares the
PyTorch function at positions `3..5` with:

```python
mlx.nn.RoPE(16, traditional=False, base=10000)(values, offset=3)
```

The test uses relative and absolute tolerance `1e-6`.

## Verification

```text
PYTHONPATH=. uv run --frozen pytest -q tests/test_flint_torch.py
4 passed, 1 skipped

uv run --frozen ruff check mlx_gptq/flint_torch.py tests/test_flint_torch.py
passed

uv run --frozen ruff format --check mlx_gptq/flint_torch.py tests/test_flint_torch.py
passed

git diff --check
passed
```

## Non-claims

- This repair does not recompute or replace the already published 256-expert
  MXFP4 checkpoint.
- It does not establish how much the previous rotary layout affected that
  artifact; a controlled recalibration is required for that claim.
- It does not run a full calibration or select a 192-expert mask.
