"""Build fixed calibration windows while preserving a protected corpus prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def covering_windows(ids: np.ndarray, length: int) -> tuple[list[np.ndarray], list[int]]:
    if len(ids) < length:
        raise ValueError("protected corpus is shorter than one window")
    starts = list(range(0, len(ids) - length + 1, length))
    if starts[-1] + length < len(ids):
        starts.append(len(ids) - length)
    return [ids[start : start + length] for start in starts], starts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True, type=Path)
    parser.add_argument("--protected-prefix", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.receipt.exists():
        parser.error("output paths must not already exist")

    from transformers import AutoTokenizer

    combined_bytes = args.combined.read_bytes()
    protected_bytes = args.protected_prefix.read_bytes()
    combined = combined_bytes.decode()
    protected = protected_bytes.decode()
    prefix = protected.rstrip() + "\n\n"
    if not combined.startswith(prefix):
        raise RuntimeError("combined corpus does not begin with the protected corpus")
    agentic = combined[len(prefix) :]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    protected_ids = np.asarray(
        tokenizer(protected, add_special_tokens=False).input_ids, dtype=np.int64
    )
    agentic_ids = np.asarray(
        tokenizer(agentic, add_special_tokens=False).input_ids, dtype=np.int64
    )
    if tokenizer.eos_token_id is not None:
        protected_ids = np.append(protected_ids, tokenizer.eos_token_id)
        agentic_ids = np.append(agentic_ids, tokenizer.eos_token_id)

    protected_windows, protected_starts = covering_windows(
        protected_ids, args.sequence_length
    )
    agentic_needed = args.samples - len(protected_windows)
    if agentic_needed < 0:
        raise RuntimeError("protected corpus alone exceeds the requested sample count")
    agentic_disjoint = len(agentic_ids) // args.sequence_length
    if agentic_disjoint < agentic_needed:
        raise RuntimeError(
            f"need {agentic_needed} agentic windows but only {agentic_disjoint} exist"
        )

    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.permutation(agentic_disjoint)[:agentic_needed].tolist())
    agentic_starts = [index * args.sequence_length for index in chosen]
    agentic_windows = [
        agentic_ids[start : start + args.sequence_length] for start in agentic_starts
    ]
    array = np.stack([*protected_windows, *agentic_windows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)

    array_bytes = args.output.read_bytes()
    receipt = {
        "algorithm": "protected-prefix-plus-disjoint-agentic-windows-v1",
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "output": {
            "path": str(args.output.resolve()),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": len(array_bytes),
            "sha256": digest(array_bytes),
            "raw_token_bytes_sha256": digest(array.tobytes(order="C")),
        },
        "protected": {
            "path": str(args.protected_prefix.resolve()),
            "sha256": digest(protected_bytes),
            "tokens": len(protected_ids),
            "windows": len(protected_windows),
            "window_starts": protected_starts,
            "all_tokens_covered": True,
            "policy": "never drop; final tail window may overlap",
        },
        "agentic": {
            "source": str(args.combined.resolve()),
            "combined_sha256": digest(combined_bytes),
            "suffix_tokens": len(agentic_ids),
            "available_disjoint_windows": agentic_disjoint,
            "selected_disjoint_windows": len(agentic_windows),
            "selected_window_indices": chosen,
            "policy": "capacity reduction applies only to this stratum",
        },
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
