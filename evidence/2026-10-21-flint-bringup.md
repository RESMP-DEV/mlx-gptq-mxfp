# Recursive Flint MXFP4 bring-up receipt

Date: 2026-10-21
Agent: Codex

## Source

- Base model: `RecursiveSI/flint`
- Revision: `2a247be0c0aec895ccb0ac7ca92229ec6df37288`
- License: Apache-2.0; public and ungated.
- Architecture: `FlintForCausalLM`, model type `flint`.
- Size: 39,785,287,456 parameters; 79,570,574,912 checkpoint bytes.
- Official Apple reference: `RecursiveSI/flint-int4` at revision
  `3b93f9053b530488405ae9fca51d4900d6ce21f0`. It is group-64 affine INT4 and
  is not an acceptance artifact for this campaign.

## Implementation

- Added an eager PyTorch calibration skeleton with exact sigmoid top-6 routing,
  ungated shared expert, per-head learned attention sinks, QK RMSNorm,
  RoPE/no-RoPE layer pattern, and 513-token sliding attention.
- The skeleton exposes routed experts as per-expert `F.linear` calls over 3D
  parameters, so the existing activation-Hessian capture engine records only
  tokens actually routed to each expert.
- Added the corresponding MLX-LM adapter in `RESMP-DEV/mlx-lm` at commit
  `428b496` (`feat: add Flint model support`).

## Calibration matrix

- Source corpus: `RESMP-DEV/ptq-calibration-corpus` at revision
  `ffa2e74b9361d573167373140f10dfd30c69a327`.
- Model-neutral input: `calibration.txt`, SHA-256
  `244a35f5289efad6397342a6782350e464a5f180d8406d2969200722d7da5fb9`.
- Tokenizer: `RecursiveSI/flint` at revision
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`.
- Tokenizer JSON SHA-256:
  `a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68`.
- Output: 256 windows of 2,048 int64 tokens.
- Output NPY SHA-256:
  `240415002e578a293c92f81af897d34279f2dee04e5c48b233a6d3205a143f77`.
- Raw token-byte SHA-256:
  `64c799b582a0a2461b0c06ca3fa2be270caeb921791ac9fe587475c7649c7ab4`.
- Local receipt:
  `build/flint-calibration/flint/calibration_256x2048.receipt.json`.

## Validation

- `python -m pytest -q tests/test_flint_torch.py` passed 3 tests.
- `ruff check tests/test_flint_torch.py mlx_gptq/calibrate.py` passed.
- Live config resolution built 49 layers; layer 2 routed expert shape was
  `(256, 512, 2048)`.
- MLX-LM adapter tests passed 3 tests and adapter commit `428b496` is pushed.

## Active transfers

- Base BF16: `/Users/kearm/models/RecursiveSI-flint-BF16-2a247be`.
- Official INT4 reference:
  `/Users/kearm/models/RecursiveSI-flint-int4-3b93f9`.
- Both use persistent local LaunchAgents and the HF xet transport. B550 also
  retains its own resumable BF16 download.

## Non-claims

- GPTQ calibration, MLX packing, output parity, performance, and Hugging Face
  publication are not complete yet.
- The official INT4 model is reference-only and will not be republished as the
  owned MXFP4 result.
