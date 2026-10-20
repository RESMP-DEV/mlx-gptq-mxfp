# DFlash2 QAT comparison correction

Date: 2026-10-20
Agent: Codex

## Decision corrected

The earlier statement that QAT MXFP4 lost to RTN MXFP4 was not a valid
full-head QAT comparison. Two defects invalidated that interpretation.

## Defect 1: selector replay severed LoRA gradients

All prior QAT/BF16 arms used `loss_mode=selector`. In that mode the trainer
replayed the captured `selector_hidden` tensor. That tensor is the draft
transformer output captured at runtime, so it is faithful input for a
selector-only fine-tune, but it has no autograd path back to the draft weights.

Live artifact audit on B550:

| Run | adapter `b` matrices nonzero | max `b` |
|---|---:|---:|
| `qat-lr2e-6` | 0 / 46 | 0.0 |
| `bf16-lr2e-6` | 0 / 46 | 0.0 |
| `qat-hybrid-r8-fable128-full-e2` | 0 / 46 | 0.0 |

The exported QAT models therefore did not contain trained draft LoRA adapters.
The only directly trained tensor was the selector projection. The apparent
movement in 49 tensors was mostly MXFP4 grid snap, not learning.

Commit `2e92b47` adds `--replay-draft`, disables the captured hidden shortcut
when it is set, includes replay mode in the resume fingerprint, and adds a
regression test proving:

- captured-hidden replay leaves adapter `b` gradients unreachable;
- draft replay sends gradients through both `fc` and decoder LoRA modules.

Verification: 50 tests passed and 5 skipped.

A post-fix 128-cycle QAT smoke proved the live gradient path:

- checkpoint `current_parameters`: 46/46 adapter `b` matrices nonzero;
- maximum `b` magnitude: `7.0766524e-5`;
- exported best snapshot still had zero adapters because the tiny run's best
  validation equaled its initial snapshot, so checkpoint selection restored
  the initialization. This is checkpoint-selection behavior, not a dead
  gradient.

## Defect 2: RTN control used a different base

The RTN MXFP4 control was built from `promoted-blend050`
(`a573f9026cc14c008bf0c1a50b55990d0fc07b3ae728c4fdf38c1851e4e5cef4`).
The QAT arms started from `promoted-g3-g20.0625-g30.1875`
(`c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`).
Those bases differ in the 47 surfaces later quantized. Therefore:

- QAT-vs-RTN was not a matched quantization comparison;
- QAT-vs-BF16 G3 was matched by base but tested only selector-projection QAT,
  not full-head QAT.

## Corrected experiments

The corrected trainer was deployed to B550. Detached runs use full draft replay:

1. QAT MXFP4 from G3;
2. QAT MXFP4 from the RTN control's G1 base;
3. BF16 replay control from the same G1 base;
4. a faster 4,096-cycle matched QAT arm from the G1 base.

These runs were active when this correction was written. Their live acceptance
has not yet been measured. No corrected QAT artifact is promoted.

## Non-claims

- The earlier QAT live rejection remains valid only as a rejection of
  selector-projection QAT under the captured-hidden shortcut.
- It is not evidence against full-head QAT.
- The RTN control's live score remains valid for that artifact, but not as a
  matched-base QAT control.

## Corrected QAT live-gate result

The first valid full-head QAT MXFP4 run used `--replay-draft` from the RTN
control's base (`promoted-blend050`), trained for 3,313 cycles, and exported
all 46 LoRA `b` matrices as nonzero (max magnitude `0.0023674`). Its packed
MLX artifact SHA-256 is
`3b868f22912c1873b68127b079f9a9a0b2fad661b77e1ce60fba972dfd896c8c`; it loaded
as 47 native MXFP4 `QuantizedLinear` modules.

Live gate on `heldout128`:

- QAT mean acceptance: `0.6684078205914255`
- RTN MXFP4 control: `0.674705131392368`
- BF16 incumbent: `0.6803182327226996`
- QAT vs RTN delta: `-0.0062973` (30 wins / 22 ties / 76 losses, p `9.10e-6`)
- QAT vs BF16 delta: `-0.0119104` (18 wins / 21 ties / 89 losses, p `1.78e-12`)
- QAT trained vs QAT zero-adapter baseline: `-0.0009543` (p `0.402`)

Decision: rejected. With the gradient-path defect fixed and a matched base,
true QAT does **not** improve the RTN MXFP4 control. The trained adapters made
no statistically meaningful live difference over the zero-adapter variant, and
both QAT artifacts remain below the RTN control. The incumbent BF16 draft
stays champion.

Paired receipt:
`build/reports/dflash/gate-qatreplay-fast-g1-heldout128-paired.json`.

Non-claims: this corrected run was a 3,313-cycle fast arm from a 11,540-cycle
top-k capture. It does not test the full 62,026-cycle corpus or the full-vocab
CE+TV objective. It is, however, the first matched-base full-head QAT live
result.
