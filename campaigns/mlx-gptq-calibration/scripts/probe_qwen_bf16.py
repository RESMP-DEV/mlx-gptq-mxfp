#!/usr/bin/env python3
"""Run a bounded BF16 Qwen3.8 generation as a quantization control."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.prompt_json) as handle:
        sample = json.load(handle)["samples"][0]
    messages = [{"role": "user", "content": sample["input"]}]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
        preserve_thinking=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "21GiB", 1: "21GiB", "cpu": "48GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(input_device) for name, value in inputs.items()}

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            use_cache=True,
        )
    elapsed = time.monotonic() - started
    new_tokens = output[0, inputs["input_ids"].shape[-1] :].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    placement = Counter(str(device) for device in model.hf_device_map.values())
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "output_tokens": len(new_tokens),
                "tokens_per_second": len(new_tokens) / elapsed,
                "device_map_counts": placement,
                "text": text,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
