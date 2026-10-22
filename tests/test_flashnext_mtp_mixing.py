from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mlx_gptq.flashnext_mtp import (
    MixedNativeMTPPairDataset,
    NativeMTPBank,
    NativeMTPPairDataset,
)


def _write_bank(root: Path, rows: int, vocab: int = 16, width: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    logits = np.arange(rows * vocab, dtype="<f4").reshape(rows, vocab) / 7.0
    residual = np.arange(rows * width, dtype="<f4").reshape(rows, width) / 3.0
    with (root / "native-logits.f32").open("wb") as handle:
        handle.write(np.array([vocab, rows], dtype="<i4").tobytes())
        handle.write(logits.tobytes())
    with (root / "native-residual.f32").open("wb") as handle:
        handle.write(np.array([width, rows], dtype="<i4").tobytes())
        handle.write(residual.tobytes())
    argmaxes = logits.argmax(axis=1)
    lines = ["round\tposition\twindow_size\trow\tinput_token\targmax\taccepted_drafts"]
    # The input chain is the prompt-final token followed by each prior argmax.
    inputs = [rows - 1] + [int(value) for value in argmaxes[:-1]]
    for index in range(rows):
        lines.append(
            f"{index}\t{500 + index}\t1\t0\t{inputs[index]}\t{int(argmaxes[index])}\t0"
        )
    (root / "native-logits.f32.windows.tsv").write_text("\n".join(lines) + "\n")
    (root / "prompt.ids").write_text(",".join(str(i) for i in range(rows)))
    return root


def test_mixed_dataset_concatenates_banks_without_realigning_pairs(
    tmp_path: Path,
) -> None:
    first = NativeMTPBank(_write_bank(tmp_path / "a", 6))
    second = NativeMTPBank(_write_bank(tmp_path / "b", 5))
    mixed = MixedNativeMTPPairDataset([first, second])
    assert mixed.per_bank_counts() == [5, 4]
    assert len(mixed) == 9
    assert mixed.bank_index(0) == 0
    assert mixed.bank_index(4) == 0
    assert mixed.bank_index(5) == 1
    assert mixed.bank_index(8) == 1
    for index in range(len(mixed)):
        record = mixed[index]
        assert torch.isfinite(record["target_logits"]).all()
        assert torch.isfinite(record["residual"]).all()
        assert int(record["target_logits"].argmax()) == int(record["target_argmax"])
        assert int(record["source_position"]) >= 500


def test_mixed_dataset_rejects_an_empty_bank_list() -> None:
    try:
        MixedNativeMTPPairDataset([])
    except ValueError as error:
        assert "at least one bank" in str(error)
    else:
        raise AssertionError("expected a ValueError for an empty bank list")


def test_single_bank_dataset_matches_the_mixed_path(tmp_path: Path) -> None:
    bank = NativeMTPBank(_write_bank(tmp_path / "a", 6))
    single = NativeMTPPairDataset(bank)
    mixed = MixedNativeMTPPairDataset([bank])
    assert len(single) == len(mixed) == 5
    for index in range(len(single)):
        assert torch.equal(
            single[index]["target_logits"], mixed[index]["target_logits"]
        )
        assert torch.equal(single[index]["residual"], mixed[index]["residual"])


def test_negative_index_comes_from_a_different_bank(tmp_path: Path) -> None:
    first = NativeMTPBank(_write_bank(tmp_path / "a", 6))
    second = NativeMTPBank(_write_bank(tmp_path / "b", 5))
    mixed = MixedNativeMTPPairDataset([first, second])
    for index in range(len(mixed)):
        negative = mixed.negative_index(index, seed=7)
        assert 0 <= negative < len(mixed)
        assert mixed.bank_index(negative) != mixed.bank_index(index)


def test_negative_index_is_deterministic(tmp_path: Path) -> None:
    first = NativeMTPBank(_write_bank(tmp_path / "a", 6))
    second = NativeMTPBank(_write_bank(tmp_path / "b", 5))
    mixed = MixedNativeMTPPairDataset([first, second])
    for index in range(len(mixed)):
        assert mixed.negative_index(index, seed=3) == mixed.negative_index(
            index, seed=3
        )
