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

- `python -m pytest -q tests/test_flint_torch.py` passed 4 tests.
- `ruff check tests/test_flint_torch.py mlx_gptq/calibrate.py` passed.
- Live config resolution built 49 layers; layer 2 routed expert shape was
  `(256, 512, 2048)`.
- MLX-LM adapter tests passed 3 tests and adapter commit `428b496` is pushed.
- A tiny four-layer BF16 fixture completed the full path: raw checkpoint →
  MLX MXFP4 conversion → Stage-A MXFP4 GPTQ → Stage-B artifact injection →
  reload. The packed model emitted `(1, 7, 128)` logits; versus BF16, mean
  logit cosine was 0.983635, minimum cosine 0.978347, and top-1 agreement was
  5/7 on the tiny random fixture. This is format proof only, not quality
  evidence for the real checkpoint.
- The eager attention mask test verifies masked additive entries are negative
  infinity and the learned sink column is preserved.
- Mac BF16 full verification: 15/15 safetensors, 79,575,138,056 bytes, zero
  failures; manifest SHA-256
  `853ebfdb733e93a3924fd7d11e14eb1dcdafec50fee6da222249e615eea0da4d`.
- B550 BF16 full verification: same 15/15 result after repairing partial
  shards from the verified Mac copy over the ZeroTier lane.
- Official INT4 reference full verification: 5/5 safetensors, 22,568,912,600
  bytes, zero failures.

## Active transfers

- Base BF16 is complete and verified on both Mac and B550.
- Official INT4 reference is complete and verified on Mac.
- The Mac transfer agents were unloaded after successful completion.

## Calibration state

- B550 calibration is running from the verified source with
  `cuda:0,cuda:1`, batch size four, MXFP4 group 32, safe-scale-search, seed
  17, and the pinned 256×2048 matrix.
- The native Strata service was idle and unqueued. It was temporarily
  unloaded through its authenticated `POST /unload` control endpoint; a
  watcher will restore it with `POST /load` after calibration.
- Layer 0 and the first failed attempt’s completed layer artifacts resumed.
- The first real MoE layer completed all 768 expert tensors in 70.2 seconds.
- A routed-slot indexing bug was caught and repaired before the resumed run:
  top-k expert slots now map back to their originating token indices, and the
  test coverage uses top-2 routing.

## Non-claims

- GPTQ calibration, MLX packing, output parity, performance, and Hugging Face
  publication are not complete yet. The calibration state above is live
  progress, not a completed run.
- The official INT4 model is reference-only and will not be republished as the
  owned MXFP4 result.
