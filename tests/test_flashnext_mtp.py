from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from mlx_gptq.flashnext_mtp import (
    NativeMTPBank,
    NativeMTPPairDataset,
    write_training_manifest,
)


def write_bank(
    root: Path,
    *,
    rows: int = 4,
    vocab: int = 7,
    width: int = 5,
) -> list[int]:
    root.mkdir(parents=True)
    logits = np.zeros((rows, vocab), dtype="<f4")
    for row in range(rows):
        logits[row, row] = 1.0
    residual = np.arange(rows * width, dtype="<f4").reshape(rows, width)
    argmax = [int(row.argmax()) for row in logits]
    inputs = [3, *argmax[:-1]]
    with (root / "native-logits.f32").open("wb") as handle:
        handle.write(struct.pack("<ii", vocab, rows))
        handle.write(logits.tobytes())
    with (root / "native-residual.f32").open("wb") as handle:
        handle.write(struct.pack("<ii", width, rows))
        handle.write(residual.tobytes())
    with (root / "native-logits.f32.windows.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "round",
                "position",
                "window_size",
                "row",
                "input_token",
                "argmax",
                "accepted_drafts",
            ]
        )
        for index in range(rows):
            writer.writerow(
                [
                    index,
                    100 + index,
                    2,
                    0,
                    inputs[index],
                    argmax[index],
                    1,
                ]
            )
    (root / "prompt.ids").write_text(",".join(["1", "2", "3"]) + "\n")
    return argmax


def test_bank_and_two_ahead_pair_dataset(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    argmax = write_bank(root)
    bank = NativeMTPBank(root)
    assert len(bank) == 4
    assert (bank.vocab_size, bank.residual_width) == (7, 5)

    pairs = NativeMTPPairDataset(bank)
    assert len(pairs) == 3
    first = pairs[0]
    assert first["residual"].shape == (5,)
    assert first["target_logits"].shape == (7,)
    assert int(first["input_token"]) == 3
    assert int(first["target_argmax"]) == argmax[1]
    assert int(first["source_position"]) == 100
    assert int(first["teacher_position"]) == 101
    assert bool(torch.isfinite(first["residual"]).all())
    assert bool(torch.isfinite(first["target_logits"]).all())


def test_one_step_teacher_is_rejected_as_misaligned(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    argmax = write_bank(root)
    sidecar = root / "native-logits.f32.windows.tsv"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    # Keep the token chain valid but lie about row 0's actual binary argmax.
    fields = lines[1].split("\t")
    wrong_argmax = (argmax[0] + 1) % 7
    fields[5] = str(wrong_argmax)
    lines[1] = "\t".join(fields)
    fields = lines[2].split("\t")
    fields[4] = str(wrong_argmax)
    lines[2] = "\t".join(fields)
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="argmax mismatch"):
        NativeMTPBank(root)[0]


def test_header_and_chain_contracts_are_enforced(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    write_bank(root)
    logits = root / "native-logits.f32"
    logits.write_bytes(logits.read_bytes()[:-4])
    with pytest.raises(ValueError, match="byte size"):
        NativeMTPBank(root)

    root2 = tmp_path / "chain"
    write_bank(root2)
    sidecar = root2 / "native-logits.f32.windows.tsv"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    fields = lines[2].split("\t")
    fields[4] = "6"
    lines[2] = "\t".join(fields)
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input chain"):
        NativeMTPBank(root2)


def test_manifest_hashes_bank_and_records_pairing(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    write_bank(root)
    bank = NativeMTPBank(root)
    output = tmp_path / "manifest.json"
    manifest = write_training_manifest(
        bank,
        output,
        source_revision="test-revision",
        dirty=False,
    )
    assert manifest["schema"] == "flashnext-native-mtp-bank-v1"
    assert manifest["bank"]["rows"] == 4
    assert set(manifest["bank"]["sha256"]) == {
        "logits",
        "residual",
        "sidecar",
        "prompt_ids",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
