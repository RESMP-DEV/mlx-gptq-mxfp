# DFlash2 QAT train, pack, and MLX load receipt

Date: 2026-10-20
Agent: Codex
Repositories: `RESMP-DEV/mlx-gptq-mxfp` (branch `work/dflash2-calibration`, head
`884f69b`), `RESMP-DEV/dflash-mlx` (branch `work/local-mxfp4-dflash2`, head
`ee36db3`), `RESMP-DEV/mlx` (main, head `02b7abe5e`).

## Question answered

Does the corrected DFlash2 full-vocabulary trainer, the QAT prescribed-grid
exporter, the native MXFP4 packer, and the MLX draft runtime now form one
consistent, gated path from captured cycles to a loadable quantized draft?

## Yes, on a bounded smoke

The full sequence ran end to end on a prompt-disjoint split of the 896-prompt,
62,026-cycle full-logit capture:

1. Four contiguous cycles from each of three training prompt groups and one
   validation prompt group, seed 43, rank-8 LoRA, alpha 16, LR `2e-6`, hybrid
   selector plus full-vocabulary CE weight 0.1 and TV weight 0.9, device CPU.
2. BF16 control run and QAT run (`--qat-mxfp4`) with identical fingerprints
   except `qat_mxfp4`. Both exited 0, both passed the startup replay-fidelity
   gate at cosine 0.99362, and both updated all 46 current-checkpoint LoRA `b`
   matrices. The QAT run emitted 47 prescribed E2M1 code and E8M0 scale pairs.
3. `tools/export_rtn_mxfp4.py --qat-source` consumed those pairs (algorithm
   `qat-prescribed-grid`, 47 tensors). `mlx_gptq.dflash2_pack` injected all 47
   tensors and wrote a native MXFP4 `config.json`.
4. Both packed checkpoints load through the MLX draft runtime as
   `prequantized: true` with 47 `QuantizedLinear` modules, `uint32` packed
   weights, and E8M0 scales.

## Defects this work found and fixed

- The trainer wrote an unquantized selector projection next to a prescribed QAT
  grid, so the checkpoint contradicted its own metadata. Fixed in
  `RESMP-DEV/mlx-gptq-mxfp` `0d50716`.
- The packer required `config.json` inside the weights-only training output.
  It now accepts `--config-root` for the base model (`4adf11e`).
- The QAT export could reselect scales and double-quantize effective weights.
  The trainer now exports exact codes and scales, and the exporter fails closed
  when they are absent or inconsistent with the checkpoint.
- The QAT replay fidelity gate was measured on the quantized forward and
  rejected valid runs. The gate now runs on the pristine model before QAT
  installation (`6630727`).

## Non-claims

- Both arms selected the initial head because a single validation cycle is too
  small to select on, so the two packed checkpoints are byte-identical. This
  receipt proves mechanism and consistency, not QAT versus RTN quality.
- No live target parity or paired acceptance run was performed. No artifact was
  promoted or uploaded.
- CPU fake-quant dominates these wall clocks and says nothing about a CUDA
  MXFP4 kernel.
