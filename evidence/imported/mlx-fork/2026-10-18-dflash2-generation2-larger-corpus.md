# DFlash2 generation-2 distillation on a larger real corpus

Date: 2026-10-18

## Scope

This records the second distillation generation for the Qwen3.8-27B DFlash2
head. Generation 1 trained on a 1,981-cycle corpus from 24 prompts and was
promoted at 50 percent adapter strength. Generation 2 rebuilds the training
corpus from a 128-prompt session-grouped Fable agentic sample, resumes from the
promoted head, and gates the result on a larger prompt-grouped live corpus.

## Prompt corpus

Prompts come from `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces` at revision
`15ba38ac01b610cc986b728a2d55fcb8db9c2096`, read on B550 from
`/home/kearm/training/data/Fable-GPT-5.5-Distillation-Traces/data/eval-data-00001.parquet`.

The builder is `tools/build_fable_dflash_prompts.py` in
`RESMP-DEV/mlx-gptq-mxfp`. It selects one final user message per session so a
session never splits across prompt groups.

- output: `/Users/kearm/mlx/build/reports/dflash/fable-agentic-128.jsonl`
- SHA-256: `f5208aaed20f3d974ca83ac3f713afc2e1d612396e264809f57215e30ead435a`
- receipt: `/Users/kearm/mlx/build/reports/dflash/fable-agentic-128-receipt.json`
- prompts: 128, one per session, seed 31
- sources: 79 codex-personal-2026-02, 48 codex-personal-2026-04, 1 codex-personal-2026-05
- prompt characters: minimum 315, maximum 5,863, mean 2,779.7

The live held-out gate reuses the trainer's prompt-grouped validation split:

- output: `/Users/kearm/mlx/build/reports/dflash/fable128-heldout25.jsonl`
- SHA-256: `5ea2f1044b5453b049192d10136bfa1191b197ebc765b069e563a4b3125274cd`
- receipt: `/Users/kearm/mlx/build/reports/dflash/fable128-heldout25-receipt.json`
- 25 prompts, seed 31, validation fraction 0.2, disjoint from the 103 training prompts

## Capture

Capture ran on the main machine from the promoted generation-1 head against the
calibrated MXFP4 target, with top-64 teacher logits and 256 generated tokens.

- output: `/Users/kearm/mlx/build/reports/dflash/fable128-promoted-capture`
- receipt: `/Users/kearm/mlx/build/reports/dflash/fable128-promoted-capture/receipt.json`
- receipt SHA-256: `2bf8e33857f259c322bb0233e8854b90f248e901e477e019140ebfc712f2b396`
- index SHA-256: `f042a690883cddff2b41f292ea6a07b4108b6ed9f1ea626ea958136d8f0feb9c`
- prompts: 128, cycles: 11,540, elapsed 2,513.76 seconds
- mean capture acceptance: `0.6424939800685889`
- draft quantization: none; draft: the promoted 50 percent generation-1 head

Cycle counts on B550 were verified after transfer:
`/home/kearm/dflash2-distill/fable128-capture` holds 11,540 cycle directories
matching the index and receipt.

## Optimizer sweeps

### Defective first attempt

An earlier generation-2 study pair was launched with the promoted base model but
the generation-1 capture directory, because the study script derived the capture
path from its root argument. Those 24 trials are retained as
`build/reports/dflash/study-receipts/fable128-g2-gpu*-summary.json` and are
explicitly invalid as larger-data evidence; their receipts report 24 prompts and
validation prompts 6, 8, 9, and 18. The bug was fixed in commit `b8dbdbe`.

### Corrected sweep

Two independent 12-trial studies on separate RTX 3090 Ti GPUs both selected:

- rank 16, alpha 32
- predicted predecessors
- one epoch
- learning rate `6.789232192251469e-6`
- weight decay `9.9389246762676756e-5`

Both studies report held-out prefix accuracy rising from `0.7545090221570345`
to `0.7733800971436358` across 128 prompt groups, with the same 25 validation
prompts as the live held-out file.

Receipts: `build/reports/dflash/study-receipts/fable128-real-g2-gpu0-summary.json`
and `...-gpu1-summary.json`.

## Full generation-2 training run

Trained from the promoted head on all 11,540 cycles with the corrected
best configuration:

- B550 output: `/home/kearm/dflash2-distill/g2-r16-fable128-full`
- adapters SHA-256: `531a0bc502bb392449e68b0c9764905ce04e45413e93e9ed2df548135207f03d`
- selector projection SHA-256: `4fb2c16b8b6ac071d1a38d1bbaccee152f0527a85ef92c5adbc4d15fd7c5b86b`
- steps: 9,292, elapsed 72.55 seconds on one RTX 3090 Ti
- held-out prefix accuracy: `0.7883303708718341`, from `0.7545090221570345`
- held-out top-1 accuracy: `0.8699214167468701`
- trainable parameters: 10,690,560

### Weights and Biases logging

The trainer gained offline W&B logging in commits `7d6d03f` and `590bc54`, and
the study script passes per-trial run names and tags. B550 runs offline only;
runs are synced from the main machine, which owns the W&B credentials.

- generation-1 two-epoch run: run `l7xn5esk`, project `dflash2-distillation`,
  entity `retis_labs`, synced from
  `build/reports/dflash/wandb-runs/r16-predicted-e2-f0a6ec6-l7xn5esk`
- generation-2 full run: run `1j2xm6qr`, synced from
  `build/reports/dflash/wandb-runs/g2-r16-fable128-full-1j2xm6qr`

Remote history JSON for the generation-1 run is stored at
`build/reports/dflash/wandb-runs/r16-predicted-e2-f0a6ec6-remote-history.json`.

## Epoch measurement

A two-epoch run on the generation-1 corpus was measured to answer how many
epochs are justified. Held-out prompt-grouped prefix accuracy was:

- epoch 0: `0.8383458662301975`
- epoch 1: `0.8129699264926121`

Training loss continued to fall across the second epoch, from a mean of
`0.5487833622107848` to `0.34963398312267`, while held-out agreement fell. The
exported checkpoint restored the best epoch, so one epoch is the better
schedule for this objective and two epochs only add cost and overfitting.

## Live gates

All runs used DFlash MLX at `dcf93d3d07aadc5a2e74fab7e44c63209d9f37a3` on the
M4 Max with the calibrated MXFP4 target and `--only-dflash` corpus runs.

### Exact parity

Every interpolated candidate passed the 256-token baseline-versus-DFlash smoke
with `token_match: true`:

- generation-1 two-epoch 0.250 blend: `0.71875`
- generation-1 two-epoch 0.375 blend: `0.72265625`
- generation-1 two-epoch 0.500 blend: `0.72265625`
- generation-1 two-epoch 0.750 blend: `0.71484375`
- generation-2 0.250 blend: `0.71875`
- generation-2 0.500 blend: `0.7109375`
- generation-2 1.000 blend: `0.70703125`

### Existing 24-prompt corpus

The generation-1 champion remains better on this corpus:

- generation-1 0.500 blend mean acceptance: `0.6712239583333334`
- generation-2 0.250 blend mean acceptance: `0.6700846354166666`
- generation-2 0.500 blend mean acceptance: `0.67041015625`

### Fable 25-prompt held-out corpus

The larger held-out corpus reverses the ordering:

- generation-1 0.500 blend mean acceptance: `0.6471875`
- generation-2 0.250 blend mean acceptance: `0.64953125`
- generation-2 0.500 blend mean acceptance: `0.64703125`

Paired generation-2 0.250 result:
`build/reports/dflash/g2-r16-fable128-blend0.250-vs-g1-heldout25.json` records
`mean_delta` `0.0023437499999999978`, with 12 wins, 5 ties, and 8 losses.

## Decision

The generation-2 0.250 interpolated head is retained locally as the
domain-matched candidate for agentic Fable prompts. It is not promoted over the
generation-1 champion, because the two live gates disagree in opposite
directions at margins below half a percentage point. Promotion requires a
candidate that improves both corpora.

Artifacts:

- generation-2 candidate:
  `/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-G2-LoRA-r16-fable128-blend0.250`
- SHA-256: `c9c802e11c98cda852792b79aa3498b365f494a73bf64bd4b9e515b29d2bc385`
- still-champion: `/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-LoRA-r16-predicted-prefix-f0a6ec6-blend0.50`
- SHA-256: `a573f9026cc14c008bf0c1a50b55990d0fc07b3ae728c4fdf38c1851e4e5cef4`

## Non-claims

- Throughput and memory values are diagnostic; thermal pressure is not controlled.
- The Fable prompts are conditioning and evaluation samples, not task-quality proof.
- Held-out Fable prompts come from the same eval shard as the training prompts.
- Captured-cycle prefix accuracy is a proxy and is not acceptance.
- Neither artifact was uploaded to Hugging Face.
- The pinned BF16 Qwen3.8-27B target was not used as the capture teacher.
