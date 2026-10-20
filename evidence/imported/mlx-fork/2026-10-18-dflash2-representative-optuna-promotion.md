# DFlash2 representative Optuna sweep and promoted interpolated head

Date: 2026-10-18

## Scope

This records the corrected DFlash2 head sweep, full-corpus training, live
acceptance measurements, and promotion decision. The target-controlled target
remains the calibrated Qwen3.8-27B MXFP4 runtime used by the existing capture;
the pinned BF16 27B checkpoint was not silently substituted as the teacher.

## Implementation and validation

Trainer repository: `RESMP-DEV/mlx-gptq-mxfp`, branch
`work/dflash2-calibration`.

- `f0a6ec6`: prompt-stratified cycle selection, runtime-aligned prefix/top-1
  metrics, teacher-versus-predicted predecessor modes, initial-state
  checkpointing, and representative study driver.
- `4fa550b`: corrected physical-GPU to CUDA-ordinal mapping.
- `cf644b3`: pruned failed trials instead of aborting the complete study.

Validation executed locally with `uv run ... pytest -q`: 26 tests passed and 3
skipped. The focused DFlash2 suite also passed on B550 with 9 tests passing.
`ruff check` and `ruff format --check` passed for the edited trainer and study
files before the commits.

## Representative sweep

Two B550 RTX 3090 Ti studies each completed 12 real trials. Each trial used
640 prompt-stratified cycles, preserved all 24 prompt groups, and held out
prompts 6, 8, 9, and 18. The objective minimized validation block prefix
error, not unary cross-entropy alone.

Both studies selected:

- LoRA rank 16;
- alpha ratio 1 for the teacher-forced candidate;
- teacher-forced predecessors;
- learning rate `2e-5`;
- weight decay `0.01`.

On the sweep split, held-out prefix accuracy improved from `65.504%` to
`73.450%`. Study summaries live at:

- `/home/kearm/dflash2-distill/studies/representative-gpu0/summary.json`;
- `/home/kearm/dflash2-distill/studies/representative-gpu1/summary.json`.

The first GPU1 launch failed before training because `CUDA_VISIBLE_DEVICES=1`
was combined with `--device cuda:1`. The mapping was corrected to expose one
physical GPU and use `cuda:0`; the resumed GPU1 study then completed normally.

## Full runs and live rejections

All full runs covered 1,981 captured cycles and one epoch.

| Candidate | Held-out prefix accuracy | Live mean acceptance | Result |
| --- | ---: | ---: | --- |
| Teacher-forced rank-16 LoRA, alpha 16 | 79.731% | 66.211% | rejected |
| Selector projection only | 76.347% | 65.999% | rejected |
| Predicted-predecessor LoRA, alpha 32 | 81.767% | 66.341% | rejected unblended |

The calibrated control mean was `66.683%`. These results show that replay
prefix accuracy can improve sharply while live acceptance regresses, so it
remains only a candidate-ranking proxy.

## Promoted interpolated artifact

The predicted-predecessor model was interpolated toward the calibrated
control. A 0.5 blend of the LoRA deltas and selector-projection delta passed
the exact 256-token target-controlled parity gate:

- token match: `true`;
- smoke acceptance: `71.484%`;
- smoke baseline throughput: `27.551` tok/s;
- smoke DFlash throughput: `49.101` tok/s.

The control was rerun under DFlash MLX revision
`dcf93d3d07aadc5a2e74fab7e44c63209d9f37a3` before promotion. The paired
24-prompt corpus result was:

- control mean acceptance: `66.683%`;
- candidate mean acceptance: `67.122%`;
- mean delta: `+0.439` percentage points;
- candidate wins/ties/losses: 13/5/6.

Matched receipt:
`/Users/kearm/mlx/build/reports/dflash/lora-r16-predicted-blend0.50-vs-calibrated-mxfp4-corpus24-matched.json`.

The 0.25 blend measured `67.090%`. The 0.75 blend matched the 0.5 mean at
`67.122%`, with 14 wins, 4 ties, and 6 losses, but 0.5 was selected as the
conservative artifact because it uses less adapter movement and has the
recorded parity gate. The unblended 1.0 model regressed to `66.341%`.

Promoted local artifact:
`/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-LoRA-r16-predicted-prefix-f0a6ec6-blend0.50`.

Model SHA-256:
`a573f9026cc14c008bf0c1a50b55990d0fc07b3ae728c4fdf38c1851e4e5cef4`.

Corpus throughput was diagnostic only: the matched control median was
`27.185` tok/s and the 0.5 blend median was `27.689` tok/s. Thermal pressure
was not controlled.

The durable architecture record is DFlash MLX commit `cd1f7ce`
(`docs/dflash2-distillation.md`, branch `work/local-mxfp4-dflash2`).

## Non-claims

This does not prove token-level parity for every corpus prompt, broad task
quality, natural-EOS behavior, long-context performance, or a controlled
thermal throughput win. The promoted artifact remains local and was not
uploaded to Hugging Face in this work unit.
