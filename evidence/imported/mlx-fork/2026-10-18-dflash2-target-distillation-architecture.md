# DFlash2 target distillation architecture correction

Date: 2026-10-18

## Scope

This records the corrected Qwen3.8-27B-to-DFlash2 training boundary and the
reusable adapter plan for Qwen3.8 Flash Next. It corrects the scope of the
Qwen3.5-0.8B DistillKit runs: those runs validated plumbing and a Qwen3.5
architecture guard, but were not DFlash2-head distillation.

## Concrete target and student

- Frozen target/verifier: `Qwen/Qwen3.8-27B`, immutable revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- BF16 target manifest: 18 model shards, approximately 54.5 GiB.
- DFlash2 head: local `z-lab/Qwen3.8-27B-DFlash2` lineage.
- Remote B550 head:
  `/home/kearm/dflash2-distill/model/model.safetensors`.
- DFlash2 model SHA-256:
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- DFlash2 config SHA-256:
  `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.
- DFlash2 shape: 1,924,404,480 BF16 parameters, five decoder layers,
  target layers 5/19/33/47/61, 25,600-wide raw target features, 5,120-wide
  draft hidden states, selector rank 256, runtime selector top-k 16, and
  vocabulary 248,320.

The target is the authority and remains frozen. The DFlash2 head is the only
student. Its output remains target-verified and speculative.

## Existing capture and training state

The existing full replay corpus on B550 contains 1,981 target-verified cycles
from 24 prompts, 256 generated tokens per prompt, and target top-64 teacher
signals:

- corpus: `/home/kearm/dflash2-distill/capture`;
- size: approximately 1.2 GB;
- capture elapsed time: 358.437 seconds;
- prior target path: the calibrated local Qwen3.8-27B MLX MXFP4 runtime;
- prior draft path: local BF16 DFlash2 with affine W4 runtime.

The original trainer was selector-only: it froze the transformer, target
projection, and both selector codebooks, and trained only the 1,310,720-parameter
selector hidden projection. Those candidates passed exact output parity but
failed the paired live-acceptance gate and were rejected. The corrected trainer
now also supports draft LoRA.

## Efficient training modes

The reusable architecture now records these modes:

- selector projection only: existing diagnostic mode;
- selector LoRA: future low-rank selector experiment;
- draft LoRA: low-rank adapters on the target projection and five decoder
  layers, the preferred next full-head mode;
- selective full-head tuning: unfreeze selected projections when adapter
  capacity is insufficient;
- QLoRA/NF4: memory-limited differentiable-base path, requiring a measured
  speed comparison;
- MXFP4 GPTQ: post-training deployment compression, not training.

Measured head shape makes draft LoRA inexpensive relative to the full student:
rank-8 adapters over the five decoder layers are about 4.44 million parameters,
rank-16 about 8.89 million, rank-32 about 17.78 million, and rank-64 about
35.55 million. Rank-32 is approximately 0.92 percent of all DFlash2 parameters.
A rank-16 adapter on the 25,600-by-5,120 target projection adds 491,520
parameters.

BF16 LoRA is the first implementation target because the frozen base remains in
a native fast dtype. QLoRA is a fallback for memory pressure; on Ampere it must
prove equal or better wall-clock speed. The existing MXFP4 GPTQ pack is not a
differentiable QLoRA implementation.

## Dataset reuse and future Flash Next port

The Fable and Infatoshi trace corpora are prompt sources for target-verified
cycle capture. They are not used to fine-tune the 27B target itself. Prompt
splits stay session-grouped; training, held-out, and full corpus captures carry
target/draft revisions, tokenizer identity, target layer IDs, top-k, block
size, generation policy, and quantization provenance.

A Qwen3.8 Flash Next target can reuse the capture and adapter-training contract
after a target adapter proves hidden-state extraction, embedding/logits
interfaces, cache rollback, verification semantics, vocabulary/tokenizer
identity, and exact native-forward parity. Different hidden width or depth
requires new feature dimensions and a new target projection.

Promotion remains unchanged in every mode: exact target-controlled output and
improved paired live acceptance. Training loss, offline agreement, compression
ratio, memory reduction, or throughput alone is not acceptance.

## Artifacts and implementation

- DFlash MLX architecture document:
  `dflash-mlx/docs/dflash2-distillation.md`.
- Branch: `work/local-mxfp4-dflash2`.
- Commit: `36859b2` (`docs: record DFlash2 distillation boundaries`).
- Remote: `RESMP-DEV/dflash-mlx`.
- LoRA trainer commit: `RESMP-DEV/mlx-gptq-mxfp`,
  branch `work/dflash2-calibration`, commit
  `3a6bf68` (`Export merged DFlash2 LoRA checkpoints cleanly`), including
  `f184d01`, `c0955d4`, `cbf66a2`, and `5ce01f2`.
- Validation: 21 tests passed and 3 skipped locally; the focused DFlash2
  suite grew to 7 tests and passed on both the Mac and B550; the full local
  suite reached 24 passed and 3 skipped.
- Real B550 BF16 LoRA smoke:
  `/home/kearm/dflash2-distill/lora-smoke-f184d01`.
  It used two real captured cycles, rank 4 over 46 linear modules, and
  3,655,680 total trainable parameters including the selector projection.
  Training completed in 6.298 seconds. The exported merged model and adapter
  files contain no non-finite floating-point tensors.

## First full LoRA campaign and rejection

A real LR probe used rank 16, 160 cycles, and two held-out prompts across
eight trials. The best probe LR was `4.7520278376003854e-5` with held-out CE
`1.254162421842111`; summary:
`/home/kearm/dflash2-distill/studies/lora-r16-lr/summary.json`.

The full run covered all 1,981 cycles and held out prompts 19 and 22:

- output: `/home/kearm/dflash2-distill/lora-r16-e1-lr4.7520278e-5-merged`;
- 1,834 optimizer steps in 98.422 seconds on one RTX 3090 Ti;
- 46 LoRA modules, rank 16, alpha 32;
- 10,690,560 trainable parameters including the selector projection;
- prompt-grouped validation CE `0.30501226877256044`;
- merged model SHA-256:
  `20d140339e0a6468ed6926ac41441202f834337ea4922ed3f1a83fe1bfd17e40`;
- adapter SHA-256:
  `8ba01e6762390548f4139d69acd40f57313fc926b8ad8d4d2e6b48d14e8a7c7c`.

The first export retained wrapper keys and was rejected by the MLX loader.
Commit `3a6bf68` now unwraps adapters after merging; the regenerated checkpoint
has the original 81 tensor keys and no `.a`, `.b`, or `.base.weight` wrapper
entries.

The corrected artifact passed exact 256-token smoke parity (`token_match: true`)
at 69.922% acceptance:
`/Users/kearm/mlx/build/reports/dflash/lora-r16-mxfp4-parity-256-v3`.
The benchmark now records per-run parity following DFlash MLX commit
`47e3483`.

The full 24-prompt paired corpus rejected it:

- calibrated MXFP4 control mean acceptance: `66.6829427083%`;
- LoRA candidate mean acceptance: `63.6067708333%`;
- delta: `-3.076171875` percentage points;
- candidate record: 1 win, 3 ties, 20 losses.

Paired receipt:
`/Users/kearm/mlx/build/reports/dflash/lora-r16-e1-vs-calibrated-mxfp4-corpus24.json`.
The artifact remains local and rejected; it was not uploaded to Hugging Face.
This result shows that BF16 LoRA makes full-head experiments extremely cheap,
but the first offline objective still does not transfer to live acceptance.

## Target transfer state

The Mac source copy passed full pinned HFD verification at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: 32/32 files and
55,586,114,863 bytes. A direct LAN seed transfer is filling
`/home/kearm/training/qwen38-27B-seed` and a watcher hard-links completed files
into `/home/kearm/training/models/Qwen3.8-27B-1d4bf0f`, then runs full HFD
verification. The original B550 HFD job was stopped because unrelated
DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next HFD jobs were consuming the
wide-area link; those unrelated jobs remain untouched.

At the time of this receipt the LAN transfer was still active. This receipt
does not claim a target transfer completion receipt, a live-acceptance win, or
Hugging Face publication.
