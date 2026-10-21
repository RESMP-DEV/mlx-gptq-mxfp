# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

"""Run one EAGLE-4 comparison measurement and emit a JSON receipt."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlx.core as mx
from dflash_mlx.engine.events import (
    PrefillCompleteEvent,
    SummaryEvent,
    TokenEvent,
    is_engine_event,
)
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.context import build_offline_runtime_context
from mlx.utils import tree_flatten
from mlx_lm import stream_generate as mlx_stream_generate


@dataclass(frozen=True)
class Args:
    method: str
    model: str
    draft: str | None
    prompt: str
    max_tokens: int
    block_tokens: int | None
    use_chat_template: bool
    verify_mode: str
    expected_ids_path: Path | None
    native_receipt_path: Path | None
    draft_quant: str | None
    output: Path


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _prompt_ids(tokenizer: Any, prompt: str, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def _parameter_bytes(model: Any) -> int:
    return sum(int(value.nbytes) for _, value in tree_flatten(model.parameters()))


def _run_native(args: Args) -> dict[str, Any]:
    from mlx_lm.utils import load

    started_ns = time.perf_counter_ns()
    model, tokenizer, _config = load(args.model, lazy=False, return_config=True)
    input_value: Any = args.prompt
    if args.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        input_value = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    generated_ids: list[int] = []
    prompt_ids = _prompt_ids(tokenizer, args.prompt, args.use_chat_template)
    last_response: Any = None
    for response in mlx_stream_generate(
        model, tokenizer, input_value, max_tokens=args.max_tokens
    ):
        generated_ids.append(int(response.token))
        last_response = response
    elapsed_us = (time.perf_counter_ns() - started_ns) / 1_000.0
    mx.clear_cache()
    if last_response is None:
        raise RuntimeError("native stream produced no response")
    prompt_tps = float(last_response.prompt_tps)
    generation_tps = float(last_response.generation_tps)
    return {
        "method": "native",
        "model_ref": args.model,
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids": generated_ids,
        "generation_tokens": len(generated_ids),
        "elapsed_us": elapsed_us,
        "prefill_us": (len(prompt_ids) / prompt_tps * 1e6) if prompt_tps > 0 else None,
        "generation_tps": generation_tps,
        "target_weight_bytes": _parameter_bytes(model),
    }


def _run_speculative(args: Args) -> dict[str, Any]:
    runtime_context = build_offline_runtime_context(
        verify_mode=args.verify_mode,
        draft_sink_size=64,
        draft_window_size=4096,
        verify_len_cap=args.max_tokens,
    )
    started_ns = time.perf_counter_ns()
    bundle = load_runtime_bundle(
        model_ref=args.model,
        draft_ref=args.draft,
        draft_quant=args.draft_quant,
        verify_config=runtime_context.verify,
    )
    first_token_us: float | None = None
    prefill: PrefillCompleteEvent | None = None
    summary: SummaryEvent | None = None
    stream = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        use_chat_template=args.use_chat_template,
        block_tokens=args.block_tokens,
        stop_token_ids=get_stop_token_ids(bundle.tokenizer),
        runtime_context=runtime_context,
    )
    try:
        for event in stream:
            if isinstance(event, PrefillCompleteEvent):
                prefill = event
            elif isinstance(event, TokenEvent):
                if first_token_us is None:
                    first_token_us = (time.perf_counter_ns() - started_ns) / 1_000.0
            elif isinstance(event, SummaryEvent):
                summary = event
            elif not is_engine_event(event):
                raise TypeError(f"Unsupported engine event: {type(event).__name__}")
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if summary is None:
        raise RuntimeError("speculative stream produced no summary")

    payload = summary.to_payload()
    elapsed_us = (time.perf_counter_ns() - started_ns) / 1_000.0
    payload.update(
        {
            "method": args.method,
            "model_ref": args.model,
            "draft_ref": args.draft,
            "elapsed_us": elapsed_us,
            "ttft_us": first_token_us,
            "target_weight_bytes": _parameter_bytes(bundle.target_model),
            "draft_weight_bytes": _parameter_bytes(bundle.draft_model),
        }
    )
    if prefill is not None:
        payload["prefill_us"] = float(prefill.prefill_us)
    generation_us = max(1.0, float(elapsed_us) - float(payload.get("prefill_us", 0.0)))
    payload["generation_tps"] = int(payload["generation_tokens"]) / (
        generation_us / 1e6
    )
    mx.clear_cache()
    return payload


def _add_parity_and_roofline(payload: dict[str, Any], args: Args) -> None:
    native_receipt: dict[str, Any] | None = None
    if args.native_receipt_path is not None:
        native_receipt = json.loads(args.native_receipt_path.read_text())
    native_ids = (
        [int(value) for value in native_receipt.get("generated_token_ids", [])]
        if native_receipt is not None
        else None
    )
    if args.expected_ids_path is not None:
        native_ids = [
            int(value) for value in json.loads(args.expected_ids_path.read_text())
        ]
    if native_ids is not None:
        payload["native_generated_token_ids"] = native_ids
        payload["exact_native_id_match"] = native_ids == payload["generated_token_ids"]
        payload["first_native_id_mismatch"] = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(native_ids, payload["generated_token_ids"], strict=False)
                )
                if left != right
            ),
            None,
        )

    target_weight_bytes = float(payload["target_weight_bytes"])
    native_tps = float(
        (native_receipt or {}).get("generation_tps")
        or payload.get("generation_tps")
        or 0.0
    )
    acceptance = sum(float(value) for value in payload.get("acceptance_history", ()))
    cycles = int(payload.get("cycles_completed", 0) or 0)
    if native_tps > 0 and target_weight_bytes > 0 and cycles > 0:
        tokens_per_cycle = acceptance / cycles + 1.0
        payload["roofline"] = {
            "target_weight_bytes": int(target_weight_bytes),
            "draft_weight_bytes": payload.get("draft_weight_bytes"),
            "native_decode_tps": native_tps,
            "effective_bandwidth_gb_s": native_tps * target_weight_bytes / 1e9,
            "mean_tokens_per_cycle": tokens_per_cycle,
            "mean_acceptance_depth": acceptance / cycles,
            "weights_only_roofline_tps": (
                tokens_per_cycle / (1.0 / native_tps)
                if "draft_weight_bytes" not in payload
                else tokens_per_cycle
                / (
                    1.0 / native_tps
                    + float(payload["draft_weight_bytes"])
                    * (acceptance / cycles)
                    / (target_weight_bytes * native_tps)
                )
            ),
        }


def parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("native", "eagle4", "dflash2"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--block-tokens", type=int)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--verify-mode", default="dflash")
    parser.add_argument("--expected-ids-path", type=Path)
    parser.add_argument("--native-receipt-path", type=Path)
    parser.add_argument("--draft-quant")
    parser.add_argument("--output", type=Path, required=True)
    raw = parser.parse_args(argv)
    return Args(
        method=raw.method,
        model=raw.model,
        draft=raw.draft,
        prompt=raw.prompt,
        max_tokens=raw.max_tokens,
        block_tokens=raw.block_tokens,
        use_chat_template=not raw.raw_prompt,
        verify_mode=raw.verify_mode,
        expected_ids_path=raw.expected_ids_path,
        native_receipt_path=raw.native_receipt_path,
        draft_quant=raw.draft_quant,
        output=raw.output,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _run_native(args) if args.method == "native" else _run_speculative(args)
    _add_parity_and_roofline(payload, args)
    payload["environment"] = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx": _package_version("mlx"),
        "mlx_lm": _package_version("mlx-lm"),
    }
    payload["non_claims"] = [
        "One prompt measures the selected operating point, not a quality benchmark.",
        "The roofline uses target weights and measured native TPS as its proxy.",
        "Exact ID match is not a broad quality claim.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.output), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
