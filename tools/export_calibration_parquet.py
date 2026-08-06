"""Export fixed PTQ calibration data as plain text and viewer-friendly Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-text", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output_text.exists():
        parser.error("output path already exists")

    from transformers import AutoTokenizer

    array = np.load(args.array)
    receipt = json.loads(args.receipt.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    if list(array.shape) != receipt["output"]["shape"]:
        raise RuntimeError("array shape does not match receipt")

    protected = receipt["protected"]["selected_window_starts"]
    general = receipt["general_agentic"]["selected_window_indices"]
    challenge = receipt["challenge_agentic"]["document_windows"]
    expected = len(protected) + len(general) + len(challenge)
    if expected != len(array):
        raise RuntimeError(f"receipt describes {expected} rows, array has {len(array)}")

    rows = []
    for sample_id, ids in enumerate(array):
        if sample_id < len(protected):
            local_index = sample_id
            stratum = "protected_v3_v5"
            document_index = None
            source_window_index = None
            source_token_start = protected[local_index]
        elif sample_id < len(protected) + len(general):
            local_index = sample_id - len(protected)
            stratum = "general_agentic"
            document_index = None
            source_window_index = general[local_index]
            source_token_start = source_window_index * array.shape[1]
        else:
            local_index = sample_id - len(protected) - len(general)
            stratum = "kernelbench_hard"
            item = challenge[local_index]
            document_index = item["document_index"]
            source_window_index = item["window_index"]
            source_token_start = item["window_start"]
        token_ids = ids.astype(np.int64).tolist()
        clean_text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).replace("\r\n", "\n").replace("\r", "\n")
        rows.append(
            {
                "sample_id": sample_id,
                "stratum": stratum,
                "input_ids": token_ids,
                "text": clean_text,
                "text_with_special_tokens": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "token_count": len(token_ids),
                "source_document_index": document_index,
                "source_window_index": source_window_index,
                "source_token_start": source_token_start,
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.tokenizer_revision,
            }
        )

    table = pa.Table.from_pylist(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd", compression_level=9)
    imatrix_text = "\n\n".join(row["text"].strip() for row in rows) + "\n"
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.write_text(imatrix_text)
    payload = args.output.read_bytes()
    text_payload = args.output_text.read_bytes()
    print(
        json.dumps(
            {
                "path": str(args.output.resolve()),
                "rows": table.num_rows,
                "columns": table.column_names,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "imatrix_text": {
                    "path": str(args.output_text.resolve()),
                    "bytes": len(text_payload),
                    "sha256": hashlib.sha256(text_payload).hexdigest(),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
