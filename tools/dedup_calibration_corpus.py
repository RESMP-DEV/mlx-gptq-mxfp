"""Combine and conservatively n-gram-deduplicate calibration text corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BLOCK_SPLIT = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
TOKEN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN.findall(normalized)


def shingles(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


@dataclass
class KeptBlock:
    source: str
    source_index: int
    text: str
    tokens: list[str]
    grams: set[tuple[str, ...]]


def deduplicate(
    paths: list[Path],
    *,
    ngram_size: int,
    exact_min_tokens: int,
    near_min_tokens: int,
    jaccard_threshold: float,
    containment_threshold: float,
):
    kept: list[KeptBlock] = []
    exact_seen: dict[str, int] = {}
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    removed = []
    source_stats = []

    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        blocks = [block.strip() for block in BLOCK_SPLIT.split(text) if block.strip()]
        stats = {
            "path": str(path.resolve()),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "blocks": len(blocks),
            "kept_blocks": 0,
            "removed_exact": 0,
            "removed_near": 0,
        }

        for source_index, block in enumerate(blocks):
            tokens = normalized_tokens(block)
            normalized = " ".join(tokens)
            exact_key = sha256_bytes(normalized.encode())
            if len(tokens) >= exact_min_tokens and exact_key in exact_seen:
                match = exact_seen[exact_key]
                removed.append(
                    {
                        "reason": "exact",
                        "source": path.name,
                        "source_index": source_index,
                        "tokens": len(tokens),
                        "matched_kept_index": match,
                        "normalized_sha256": exact_key,
                    }
                )
                stats["removed_exact"] += 1
                continue

            grams = shingles(tokens, ngram_size)
            best = None
            if len(tokens) >= near_min_tokens and grams:
                candidates: set[int] = set()
                for gram in grams:
                    candidates.update(inverted.get(gram, ()))
                for candidate_index in candidates:
                    candidate = kept[candidate_index]
                    if not candidate.grams:
                        continue
                    intersection = len(grams & candidate.grams)
                    union = len(grams | candidate.grams)
                    jaccard = intersection / union
                    containment = intersection / len(grams)
                    score = max(
                        jaccard / jaccard_threshold,
                        containment / containment_threshold,
                    )
                    if best is None or score > best[0]:
                        best = (
                            score,
                            candidate_index,
                            jaccard,
                            containment,
                        )

            if best is not None and (
                best[2] >= jaccard_threshold or best[3] >= containment_threshold
            ):
                removed.append(
                    {
                        "reason": "near",
                        "source": path.name,
                        "source_index": source_index,
                        "tokens": len(tokens),
                        "matched_kept_index": best[1],
                        "jaccard": best[2],
                        "containment": best[3],
                        "normalized_sha256": exact_key,
                    }
                )
                stats["removed_near"] += 1
                continue

            kept_index = len(kept)
            kept.append(KeptBlock(path.name, source_index, block, tokens, grams))
            stats["kept_blocks"] += 1
            if len(tokens) >= exact_min_tokens:
                exact_seen[exact_key] = kept_index
            if len(tokens) >= near_min_tokens:
                for gram in grams:
                    inverted[gram].add(kept_index)

        source_stats.append(stats)

    return kept, removed, source_stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--exact-min-tokens", type=int, default=20)
    parser.add_argument("--near-min-tokens", type=int, default=50)
    parser.add_argument("--jaccard-threshold", type=float, default=0.82)
    parser.add_argument("--containment-threshold", type=float, default=0.92)
    parser.add_argument(
        "--emit-last-source-only",
        action="store_true",
        help="compare against every input but emit only blocks retained from the final input",
    )
    args = parser.parse_args(argv)

    for path in args.inputs:
        if not path.is_file():
            parser.error(f"input is not a file: {path}")
    if args.output.exists() or args.receipt.exists():
        parser.error("output and receipt paths must not already exist")

    kept, removed, source_stats = deduplicate(
        args.inputs,
        ngram_size=args.ngram_size,
        exact_min_tokens=args.exact_min_tokens,
        near_min_tokens=args.near_min_tokens,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
    )
    emitted = kept
    if args.emit_last_source_only:
        emitted = [block for block in kept if block.source == args.inputs[-1].name]
    output_text = "\n\n".join(block.text for block in emitted) + "\n"
    output_bytes = output_text.encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)

    model_tokens = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model_tokens = len(tokenizer.encode(output_text))

    receipt = {
        "algorithm": "ordered-block-exact-plus-word-ngram-near-dedup-v1",
        "parameters": {
            "ngram_size": args.ngram_size,
            "exact_min_tokens": args.exact_min_tokens,
            "near_min_tokens": args.near_min_tokens,
            "jaccard_threshold": args.jaccard_threshold,
            "containment_threshold": args.containment_threshold,
            "comparison_normalization": "Unicode NFKC, casefold, word-or-punctuation tokens",
            "output_separator": "two LF characters",
        },
        "sources": source_stats,
        "output": {
            "path": str(args.output.resolve()),
            "bytes": len(output_bytes),
            "sha256": sha256_bytes(output_bytes),
            "blocks": len(emitted),
            "model_tokens": model_tokens,
            "full_1024_windows": None if model_tokens is None else (model_tokens - 1) // 1024,
            "full_2048_windows": None if model_tokens is None else (model_tokens - 1) // 2048,
        },
        "removed": {
            "blocks": len(removed),
            "exact": sum(item["reason"] == "exact" for item in removed),
            "near": sum(item["reason"] == "near" for item in removed),
            "normalized_tokens": sum(item["tokens"] for item in removed),
            "records": removed,
        },
        "emission_policy": (
            "retained blocks from final input only"
            if args.emit_last_source_only
            else "all retained blocks"
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt["output"], "removed": receipt["removed"] | {"records": "omitted"}}, indent=2))


if __name__ == "__main__":
    main()
