# MLX Stack Agent Guide

`CLAUDE.md` is a relative symlink to this file.

## Scope

This repository owns the calibrated MLX quantization stack: GPTQ calibration,
DFlash runtime work, Qwen campaign tooling, CUDA/MXFP probes, and their
evidence. `RESMP-DEV/mlx` remains a pure upstream MLX fork and
`RESMP-DEV/mlx-lm` remains an external runtime contract; do not move either
into this repository.

## Layout ownership

- Root `mlx_gptq/`, `tools/`, and `tests/` own the active calibration engine.
- `runtime/dflash-mlx/` owns the DFlash MLX runtime lineage.
- `campaigns/mlx-gptq-calibration/` owns reusable Qwen campaign tooling and
  dated campaign evidence.
- `cuda/mlx-cuda-mxfp/` owns the standalone CUDA probe lineage.

## Verification

Run `just quality` before publishing. It writes
`build/reports/quality/receipt.json` and checks the root engine, DFlash
runtime, campaign package, and CUDA probe. Treat a missing or stale receipt as
a failed run. GPU-only claims additionally require a dated live hardware
receipt with exact commands, device/library versions, source provenance, and
non-claims.

## Artifact provenance

Never use a community-quantized target or drafter as an acceptance baseline.
External unquantized artifacts may be inspected as provenance controls, but
quantized comparisons must come from the pinned source and this repository's
calibration/packing path. If an external quantized artifact is downloaded by
mistake, remove it from the decision set and replace the measurement with a
locally quantized artifact before publishing a verdict.

## Git and upstream policy

Publish only to RESMP-DEV remotes. Never push to or open contributions against
`ml-explore/mlx`, `ml-explore/mlx-lm`, or vendor repositories. Update imported
owned lineages with `git subtree pull` from their fetch-only RESMP-DEV remotes.
