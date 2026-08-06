# Validation

Validation performed on 2026-08-06:

- `27 passed` from the complete synthetic unit-test suite.
- Source distribution and wheel both built successfully with `uv build`.
- Tests cover native MXFP packing, MLX grid parity, GPTQ error reduction,
  fused-expert routing capture, artifact selection, and LFM ShortConv AWQ.

No private model weights or calibration samples are required by the tests.
