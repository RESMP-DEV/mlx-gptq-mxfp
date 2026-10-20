#!/usr/bin/env python3
"""Capture matched teacher-forced logits for cross-backend MLX parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--continuation-json", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force-reference", action="store_true")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().rstrip("\n")
    continuation = json.loads(Path(args.continuation_json).read_text())["text"]
    model, tokenizer = load(args.model)
    if args.force_reference:
        model.train()

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
        preserve_thinking=True,
    )
    continuation_ids = tokenizer.encode(
        continuation, add_special_tokens=False
    )[: args.tokens]
    sequence = prompt_ids + continuation_ids
    logits = model(mx.array([sequence]))
    selected = logits[0, len(prompt_ids) - 1 : len(sequence) - 1].astype(mx.float32)
    mx.eval(selected)
    logits_np = np.asarray(selected, dtype=np.float32)
    targets = np.asarray(continuation_ids, dtype=np.int32)
    top1 = logits_np.argmax(axis=-1).astype(np.int32)
    target_rank = np.sum(logits_np > logits_np[np.arange(len(targets)), targets, None], axis=-1)

    np.savez_compressed(
        args.output,
        logits=logits_np.astype(np.float16),
        targets=targets,
        top1=top1,
        target_rank=target_rank.astype(np.int32),
    )
    print(
        json.dumps(
            {
                "device": str(mx.default_device()),
                "prompt_tokens": len(prompt_ids),
                "continuation_tokens": len(continuation_ids),
                "teacher_forced_top1_accuracy": float(np.mean(top1 == targets)),
                "median_target_rank": float(np.median(target_rank)),
                "output": str(Path(args.output).resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
