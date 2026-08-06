"""Build fixed calibration windows with protected and quota-controlled strata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenize(tokenizer, path: Path) -> tuple[bytes, np.ndarray]:
    raw = path.read_bytes()
    ids = np.asarray(
        tokenizer(raw.decode(), add_special_tokens=False).input_ids, dtype=np.int64
    )
    if tokenizer.eos_token_id is not None:
        ids = np.append(ids, tokenizer.eos_token_id)
    return raw, ids


def covering_windows(ids: np.ndarray, length: int) -> tuple[list[np.ndarray], list[int]]:
    if len(ids) < length:
        raise ValueError("protected corpus is shorter than one window")
    starts = list(range(0, len(ids) - length + 1, length))
    if starts[-1] + length < len(ids):
        starts.append(len(ids) - length)
    return [ids[start : start + length] for start in starts], starts


def sampled_windows(
    ids: np.ndarray, length: int, count: int, seed: int
) -> tuple[list[np.ndarray], list[int], list[int]]:
    available = len(ids) // length
    if available < count:
        raise RuntimeError(f"need {count} windows but only {available} are available")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.permutation(available)[:count].tolist())
    starts = [index * length for index in indices]
    return [ids[start : start + length] for start in starts], starts, indices


def sampled_document_windows(
    tokenizer, text: str, marker: str, length: int, count: int, seed: int
) -> tuple[list[np.ndarray], list[dict]]:
    parts = text.split(marker)
    documents = [marker + part for part in parts[1:] if part.strip()]
    if parts[0].strip() or not documents:
        raise RuntimeError("challenge split marker did not cleanly delimit documents")

    tokenized = [
        np.asarray(
            tokenizer(document, add_special_tokens=False).input_ids, dtype=np.int64
        )
        for document in documents
    ]
    available = [(doc, index) for doc, ids in enumerate(tokenized) for index in range(len(ids) // length)]
    if len(available) < count:
        raise RuntimeError(f"need {count} document windows but only {len(available)} are available")
    if len(tokenized) > count:
        raise RuntimeError("challenge window count cannot cover every document")
    if any(len(ids) < length for ids in tokenized):
        raise RuntimeError("a challenge document is shorter than one window")

    rng = np.random.default_rng(seed)
    chosen = {(doc, int(rng.integers(len(ids) // length))) for doc, ids in enumerate(tokenized)}
    remaining = [item for item in available if item not in chosen]
    rng.shuffle(remaining)
    chosen.update(remaining[: count - len(chosen)])
    ordered = sorted(chosen)
    windows = [tokenized[doc][index * length : (index + 1) * length] for doc, index in ordered]
    records = [
        {
            "document_index": doc,
            "window_index": index,
            "window_start": index * length,
        }
        for doc, index in ordered
    ]
    return windows, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", required=True, type=Path)
    parser.add_argument("--general-agentic", required=True, type=Path)
    parser.add_argument(
        "--general-protected-prefix",
        type=Path,
        help="remove this exact protected prefix from the general-agentic input",
    )
    parser.add_argument("--challenge-agentic", required=True, type=Path)
    parser.add_argument(
        "--challenge-split-marker",
        help="guarantee at least one selected window per marker-delimited document",
    )
    parser.add_argument("--general-windows", type=int, default=160)
    parser.add_argument("--challenge-windows", type=int, default=40)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.receipt.exists():
        parser.error("output paths must not already exist")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    protected_raw, protected_ids = tokenize(tokenizer, args.protected)
    if args.general_protected_prefix:
        combined_raw = args.general_agentic.read_bytes()
        general_prefix_raw = args.general_protected_prefix.read_bytes()
        prefix = general_prefix_raw.rstrip() + b"\n\n"
        if not combined_raw.startswith(prefix):
            raise RuntimeError("general-agentic input does not begin with protected prefix")
        general_raw = combined_raw[len(prefix) :]
        general_ids = np.asarray(
            tokenizer(general_raw.decode(), add_special_tokens=False).input_ids,
            dtype=np.int64,
        )
        if tokenizer.eos_token_id is not None:
            general_ids = np.append(general_ids, tokenizer.eos_token_id)
    else:
        general_raw, general_ids = tokenize(tokenizer, args.general_agentic)
    challenge_raw, challenge_ids = tokenize(tokenizer, args.challenge_agentic)

    protected_windows, protected_starts = covering_windows(
        protected_ids, args.sequence_length
    )
    general_windows, general_starts, general_indices = sampled_windows(
        general_ids, args.sequence_length, args.general_windows, args.seed
    )
    challenge_records = None
    if args.challenge_split_marker:
        challenge_windows, challenge_records = sampled_document_windows(
            tokenizer,
            challenge_raw.decode(),
            args.challenge_split_marker,
            args.sequence_length,
            args.challenge_windows,
            args.seed + 1,
        )
        challenge_starts = []
        challenge_indices = []
    else:
        challenge_windows, challenge_starts, challenge_indices = sampled_windows(
            challenge_ids, args.sequence_length, args.challenge_windows, args.seed + 1
        )
    array = np.stack([*protected_windows, *general_windows, *challenge_windows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)

    def stratum(path, raw, ids, windows, starts, indices=None):
        value = {
            "path": str(path.resolve()),
            "sha256": digest(raw),
            "tokens": len(ids),
            "available_disjoint_windows": len(ids) // args.sequence_length,
            "selected_windows": len(windows),
            "selected_window_starts": starts,
        }
        if indices is not None:
            value["selected_window_indices"] = indices
        return value

    output_raw = args.output.read_bytes()
    receipt = {
        "algorithm": "protected-plus-quota-controlled-agentic-strata-v1",
        "purpose": "post-training quantization activation calibration only; never fine-tuning",
        "samples": len(array),
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "output": {
            "path": str(args.output.resolve()),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": len(output_raw),
            "sha256": digest(output_raw),
            "raw_token_bytes_sha256": digest(array.tobytes(order="C")),
        },
        "protected": stratum(
            args.protected, protected_raw, protected_ids, protected_windows, protected_starts
        )
        | {"all_tokens_covered": True, "policy": "never drop; final tail may overlap"},
        "general_agentic": stratum(
            args.general_agentic,
            general_raw,
            general_ids,
            general_windows,
            general_starts,
            general_indices,
        )
        | {
            "protected_prefix_removed": (
                str(args.general_protected_prefix.resolve())
                if args.general_protected_prefix
                else None
            )
        },
        "challenge_agentic": stratum(
            args.challenge_agentic,
            challenge_raw,
            challenge_ids,
            challenge_windows,
            challenge_starts,
            challenge_indices,
        )
        | {
            "split_marker": args.challenge_split_marker,
            "document_windows": challenge_records,
            "all_documents_covered": challenge_records is not None,
        },
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
