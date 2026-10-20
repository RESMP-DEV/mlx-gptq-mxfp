#!/usr/bin/env python3
"""Run a bounded, directly comparable Qwen MLX generation probe."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--worker-thread", action="store_true")
    args = parser.parse_args()

    prompt = sys.stdin.read()

    def run_probe() -> dict:
        model, tokenizer = load(args.model)
        if args.force_reference:
            model.train()
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
            preserve_thinking=True,
        )

        mx.random.seed(args.seed)
        started = time.time()
        text = generate(
            model,
            tokenizer,
            token_ids,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                temp=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=0.0,
            ),
            logits_processors=make_logits_processors(
                repetition_penalty=1.0,
                presence_penalty=0.0,
            ),
            verbose=False,
        )
        return {
            "elapsed_seconds": time.time() - started,
            "prompt_tokens": len(token_ids),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "worker_thread": args.worker_thread,
            "closed_think": "</think>" in text,
            "text": text,
        }

    if args.worker_thread:
        results: list[dict] = []
        thread = threading.Thread(target=lambda: results.append(run_probe()))
        thread.start()
        thread.join()
        result = results[0]
    else:
        result = run_probe()
    text = result["text"]
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False))
    print(
        json.dumps(
            {
                **{key: value for key, value in result.items() if key != "text"},
                "chars": len(text),
                "head": text[:500],
                "tail": text[-1000:],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
