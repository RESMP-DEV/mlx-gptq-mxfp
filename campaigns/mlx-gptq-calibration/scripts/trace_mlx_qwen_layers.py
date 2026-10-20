#!/usr/bin/env python3
"""Capture Qwen3.5-family hidden states for cross-backend parity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    model, tokenizer = load(args.model)
    if args.force_reference:
        model.train()

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=args.reasoning_effort,
        preserve_thinking=True,
    )
    inputs = mx.array([token_ids])
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(inputs)
    fa_mask = create_attention_mask(hidden, None)
    ssm_mask = create_ssm_mask(hidden, None)

    vectors: dict[str, np.ndarray] = {
        "tokens": np.asarray(token_ids, dtype=np.int32),
        "embedding_last": np.asarray(hidden[0, -1].astype(mx.float32)),
    }
    summaries = []
    for index, layer in enumerate(text_model.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
        mx.eval(hidden)
        vector = np.asarray(hidden[0, -1].astype(mx.float32))
        vectors[f"layer_{index:04d}"] = vector
        summaries.append(
            {
                "layer": index,
                "kind": "linear" if layer.is_linear else "attention",
                "mean": float(vector.mean()),
                "std": float(vector.std()),
                "l2": float(np.linalg.norm(vector)),
                "min": float(vector.min()),
                "max": float(vector.max()),
            }
        )

    hidden = text_model.norm(hidden)
    logits = model.language_model.lm_head(hidden)
    mx.eval(logits)
    final_hidden = np.asarray(hidden[0, -1].astype(mx.float32))
    final_logits = np.asarray(logits[0, -1].astype(mx.float32))
    vectors["final_hidden"] = final_hidden
    vectors["final_logits"] = final_logits
    np.savez_compressed(args.output, **vectors)

    top = np.argpartition(final_logits, -20)[-20:]
    top = top[np.argsort(final_logits[top])[::-1]]
    print(
        json.dumps(
            {
                "device": str(mx.default_device()),
                "prompt_tokens": len(token_ids),
                "force_reference": args.force_reference,
                "layers": summaries,
                "top_logits": [
                    {
                        "token_id": int(token),
                        "text": tokenizer.decode([int(token)]),
                        "logit": float(final_logits[token]),
                    }
                    for token in top
                ],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
