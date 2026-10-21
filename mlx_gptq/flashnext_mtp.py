"""Validated loaders for Strata's native Flash-Next MTP teacher banks.

Strata emits two row-aligned matrices and one TSV sidecar:

* full-vocabulary verifier logits, prefixed by ``[vocab, rows]`` int32 values;
* final verifier residual rows, prefixed by ``[hc * hidden, rows]``;
* the accepted row's position, input token, target argmax, and verify-window key.

The native Qwen4Exp MTP cell is two-ahead, not one-ahead.  Cell ``i`` pairs the
target residual at position ``i`` with the token at ``i + 1`` and predicts the
token at ``i + 2``.  Consequently :class:`NativeMTPPairDataset` pairs source
row ``i`` with teacher logits from row ``i + 1``.  Using row ``i``'s logits as
the target would silently train the main decoder's one-step distribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SIDECAR_COLUMNS = (
    "round",
    "position",
    "window_size",
    "row",
    "input_token",
    "argmax",
    "accepted_drafts",
)


@dataclass(frozen=True)
class BankRow:
    """One accepted verifier row's alignment metadata."""

    round_index: int
    position: int
    window_size: int
    row_index: int
    input_token: int
    argmax: int
    accepted_drafts: int


def _matrix(path: Path) -> tuple[np.memmap, int, int]:
    """Open a headed little-endian float32 matrix and validate its byte size."""
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"{path}: matrix is shorter than its header")
    with path.open("rb") as handle:
        width, rows = struct.unpack("<ii", handle.read(8))
    if width <= 0 or rows <= 0:
        raise ValueError(f"{path}: invalid matrix dimensions {width}x{rows}")
    expected = 8 + int(width) * int(rows) * 4
    if size != expected:
        raise ValueError(f"{path}: byte size {size}, expected {expected}")
    matrix = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, width))
    return matrix, width, rows


def _token_ids(path: Path) -> list[int]:
    values = [
        int(value)
        for value in path.read_text(encoding="utf-8").replace(",", " ").split()
    ]
    if not values:
        raise ValueError(f"{path}: prompt token file is empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{path}: prompt contains a negative token id")
    return values


def sha256(path: Path) -> str:
    """Hash an artifact without loading the complete bank into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeMTPBank:
    """Strict, memory-mapped reader for one Strata native teacher bank."""

    def __init__(
        self,
        root: Path,
        *,
        logits_name: str = "native-logits.f32",
        residual_name: str = "native-residual.f32",
    ):
        self.root = Path(root)
        self.logits_path = self.root / logits_name
        self.residual_path = self.root / residual_name
        self.sidecar_path = self.logits_path.with_suffix(".f32.windows.tsv")
        self.prompt_path = self.root / "prompt.ids"
        for path in (
            self.logits_path,
            self.residual_path,
            self.sidecar_path,
            self.prompt_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.logits, self.vocab_size, self.rows = _matrix(self.logits_path)
        self.residual, self.residual_width, residual_rows = _matrix(self.residual_path)
        if residual_rows != self.rows:
            raise ValueError(
                f"{self.residual_path}: residual rows {residual_rows} do not "
                f"match logits rows {self.rows}"
            )
        self.prompt_ids = _token_ids(self.prompt_path)
        self.rows_metadata = self._read_sidecar()
        self._validate_chain()

    def _read_sidecar(self) -> list[BankRow]:
        records: list[BankRow] = []
        with self.sidecar_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != SIDECAR_COLUMNS:
                raise ValueError(
                    f"{self.sidecar_path}: sidecar columns must be {SIDECAR_COLUMNS}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    record = BankRow(
                        round_index=int(row["round"]),
                        position=int(row["position"]),
                        window_size=int(row["window_size"]),
                        row_index=int(row["row"]),
                        input_token=int(row["input_token"]),
                        argmax=int(row["argmax"]),
                        accepted_drafts=int(row["accepted_drafts"]),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{self.sidecar_path}:{line_number}: malformed row: {error}"
                    ) from error
                records.append(record)
        if len(records) != self.rows:
            raise ValueError(
                f"{self.sidecar_path}: {len(records)} rows, expected {self.rows}"
            )
        return records

    def _validate_chain(self) -> None:
        keys = [(record.round_index, record.row_index) for record in self.rows_metadata]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.sidecar_path}: duplicate round/row keys")
        positions = [record.position for record in self.rows_metadata]
        if positions != list(range(positions[0], positions[0] + self.rows)):
            raise ValueError(f"{self.sidecar_path}: positions are not contiguous")
        for record in self.rows_metadata:
            if not 0 <= record.input_token < self.vocab_size:
                raise ValueError(f"{self.sidecar_path}: input token out of range")
            if not 0 <= record.argmax < self.vocab_size:
                raise ValueError(f"{self.sidecar_path}: argmax out of range")
            if not 0 <= record.row_index <= record.accepted_drafts:
                raise ValueError(
                    f"{self.sidecar_path}: emitted row follows a rejected draft"
                )

        inputs = [record.input_token for record in self.rows_metadata]
        argmaxes = [record.argmax for record in self.rows_metadata]
        expected_inputs = [self.prompt_ids[-1], *argmaxes[:-1]]
        if inputs != expected_inputs:
            raise ValueError(
                "teacher input chain does not match prompt-final token followed "
                "by prior target argmax values"
            )

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.rows:
            raise IndexError(index)
        record = self.rows_metadata[index]
        logits = torch.from_numpy(np.array(self.logits[index], copy=True)).float()
        residual = torch.from_numpy(np.array(self.residual[index], copy=True)).float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(f"{self.logits_path}: non-finite logits at row {index}")
        if not bool(torch.isfinite(residual).all()):
            raise ValueError(
                f"{self.residual_path}: non-finite residual at row {index}"
            )
        if int(logits.argmax()) != record.argmax:
            raise ValueError(
                f"{self.logits_path}: argmax mismatch at row {index}: "
                f"{int(logits.argmax())} != {record.argmax}"
            )
        return {
            "residual": residual,
            "input_token": torch.tensor(record.input_token, dtype=torch.long),
            "target_logits": logits,
            "target_argmax": torch.tensor(record.argmax, dtype=torch.long),
            "position": torch.tensor(record.position, dtype=torch.long),
            "round_index": torch.tensor(record.round_index, dtype=torch.long),
            "row_index": torch.tensor(record.row_index, dtype=torch.long),
        }

    def summary(self, *, hashes: bool = False) -> dict[str, Any]:
        """Return a sanitized bank summary for training receipts."""
        result: dict[str, Any] = {
            "root": str(self.root),
            "rows": self.rows,
            "vocab_size": self.vocab_size,
            "residual_width": self.residual_width,
            "prompt_tokens": len(self.prompt_ids),
            "positions": [
                self.rows_metadata[0].position,
                self.rows_metadata[-1].position,
            ],
        }
        if hashes:
            result["sha256"] = {
                "logits": sha256(self.logits_path),
                "residual": sha256(self.residual_path),
                "sidecar": sha256(self.sidecar_path),
                "prompt_ids": sha256(self.prompt_path),
            }
        return result


class NativeMTPPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Pair each MTP input cell with the following verifier teacher row."""

    def __init__(self, bank: NativeMTPBank):
        if len(bank) < 2:
            raise ValueError("native MTP training requires at least two bank rows")
        self.bank = bank

    def __len__(self) -> int:
        return len(self.bank) - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        source = self.bank[index]
        teacher = self.bank[index + 1]
        return {
            "residual": source["residual"],
            "input_token": source["input_token"],
            "source_position": source["position"],
            "target_logits": teacher["target_logits"],
            "target_argmax": teacher["target_argmax"],
            "teacher_position": teacher["position"],
        }


class MixedNativeMTPPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Concatenate several native MTP banks into one mixed training set.

    Every bank keeps its own two-ahead pairing, so mixing never changes a
    pair's alignment.  The mix is what keeps the head from specializing to a
    single prompt: a bank drawn from one source contributes only its own
    share of the rows, so a head that memorizes one bank's residuals pays for
    it with the rows it did not train on.
    """

    def __init__(self, banks: list[NativeMTPBank]):
        if not banks:
            raise ValueError("a mixed MTP dataset needs at least one bank")
        self.datasets = [NativeMTPPairDataset(bank) for bank in banks]
        self.offsets: list[int] = []
        total = 0
        for dataset in self.datasets:
            self.offsets.append(total)
            total += len(dataset)
        self.length = total

    def __len__(self) -> int:
        return self.length

    def bank_index(self, index: int) -> int:
        """Return the bank that owns a row index."""
        owner = 0
        for position, offset in enumerate(self.offsets):
            if index >= offset:
                owner = position
        return owner

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        owner = self.bank_index(index)
        offset = self.offsets[owner]
        record = dict(self.datasets[owner][index - offset])
        record["bank_index"] = torch.tensor(owner, dtype=torch.long)
        return record

    def per_bank_counts(self) -> list[int]:
        """Return the number of pairs each bank contributes."""
        return [len(dataset) for dataset in self.datasets]


def write_training_manifest(
    bank: NativeMTPBank,
    output: Path,
    *,
    source_revision: str,
    dirty: bool,
) -> dict[str, Any]:
    """Write the strict pre-training manifest consumed by a future MTP trainer."""
    manifest = {
        "schema": "flashnext-native-mtp-bank-v1",
        "bank": bank.summary(hashes=True),
        "source_revision": source_revision,
        "source_dirty": dirty,
        "pairing": {
            "source": "verifier residual row i plus input token at i+1",
            "teacher": "full-vocabulary verifier logits row i+1",
            "reason": "native Qwen4Exp MTP predicts position i+2",
        },
        "non_claims": [
            "This manifest validates data alignment only; it does not train or "
            "qualify an MTP candidate.",
            "Full-vocabulary logits are the frozen teacher signal, not student output.",
        ],
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
