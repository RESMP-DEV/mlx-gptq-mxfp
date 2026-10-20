"""Measure one Qwen3.8-27B variant on an A100: held-out perplexity + throughput.

Held-out perplexity uses the agentic-coding long-context corpus that is provably
zero-overlap (20-word shingles) with the calibration text, so it is a fair
matched control across BF16 / FP8 / MXFP4A16 variants. Throughput is a fixed
deterministic decode under vLLM.

This script runs INSIDE the Modal GPU container. It never mutates the model; it
only reads. Every claim in the receipt is bound to file hashes and live GPU
identity, not to a directory name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


class BenchError(ValueError):
    pass


def recorded_at() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    try:
        return {
            "vllm": importlib.metadata.version("vllm"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise BenchError(f"required package missing: {exc}") from exc


def gpu_state() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise BenchError("CUDA is not available; benchmark must run on a GPU")
    return [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]


def window_plan(
    total_tokens: int,
    seq_len: int,
    stride: int,
    max_chunks: int,
) -> list[tuple[int, int, int]]:
    """Return `(begin, end, scored_tokens)` for causal sliding-window scoring.

    Hugging Face causal models shift labels internally, so a label at position
    ``i`` is predicted from the prefix before it. Position zero therefore has no
    predecessor and is never scored in a window. Every later token is scored
    exactly once: in the first window if it is below ``seq_len``, otherwise in
    the first window whose carried context reaches its predecessor.
    """
    if total_tokens <= 0:
        raise BenchError(f"token stream is empty: {total_tokens}")
    if seq_len <= 0:
        raise BenchError(f"seq_len must be positive: {seq_len}")
    if stride <= 0:
        raise BenchError(f"stride must be positive: {stride}")
    if stride > seq_len:
        raise BenchError(f"stride ({stride}) must not exceed seq_len ({seq_len})")
    if max_chunks <= 0:
        raise BenchError(f"max_chunks must be positive: {max_chunks}")

    windows: list[tuple[int, int, int]] = []
    begin = 0
    previous_end = 0
    while begin < total_tokens and len(windows) < max_chunks:
        end = min(begin + seq_len, total_tokens)
        if previous_end < begin:
            raise BenchError(
                f"window plan leaves an unscored gap: previous_end={previous_end} begin={begin}"
            )
        scored = end - max(begin + 1, previous_end)
        windows.append((begin, end, scored))
        previous_end = end
        if end == total_tokens:
            break
        begin += stride
    return windows


def causal_score_slices(length: int, scored_tokens: int) -> tuple[slice, slice]:
    """Return logits/label slices for the final causal-scored positions."""
    if length <= 0:
        raise BenchError(f"window length must be positive: {length}")
    if not 0 <= scored_tokens < length:
        raise BenchError(f"scored token count must be in [0, {length - 1}]: {scored_tokens}")
    return (
        slice(length - scored_tokens - 1, length - 1),
        slice(length - scored_tokens, length),
    )


def model_identity(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise BenchError(f"model config.json missing: {config_path}")
    config = json.loads(config_path.read_text())
    index_path = model / "model.safetensors.index.json"
    shards: list[str]
    if index_path.is_file():
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    else:
        shards = [p.name for p in sorted(model.glob("*.safetensors"))]
    if not shards:
        raise BenchError(f"no safetensors shards under {model}")
    shard_hashes = {}
    for name in shards:
        shard_hashes[name] = sha256_file(model / name)
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quantization_config_present": "quantization_config" in config,
        "quantization_format": (config.get("quantization_config") or {}).get("format"),
        "config_sha256": sha256_file(config_path),
        "shard_count": len(shards),
        "shard_sha256": shard_hashes,
    }


def perplexity(
    model: Path,
    holdout_txt: Path,
    *,
    seq_len: int,
    stride: int,
    max_chunks: int,
    dtype: str,
) -> dict[str, Any]:
    """Sliding-window negative log-likelihood perplexity over held-out text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    text = holdout_txt.read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(ids)
    if total <= seq_len:
        raise BenchError(f"holdout too short for seq_len={seq_len}: {total} tokens")

    torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
    t0 = time.perf_counter()
    loaded = AutoModelForCausalLM.from_pretrained(
        str(model),
        local_files_only=True,
        dtype=torch_dtype,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    load_seconds = time.perf_counter() - t0
    loaded.eval()

    windows = window_plan(
        total,
        seq_len=seq_len,
        stride=stride,
        max_chunks=max_chunks,
    )
    nll = 0.0
    counted = 0
    with torch.no_grad():
        for begin, end, scored in windows:
            input_ids = torch.tensor([ids[begin:end]], device="cuda:0")
            out = loaded(input_ids)
            logits_slice, labels_slice = causal_score_slices(end - begin, scored)
            loss_sum = torch.nn.functional.cross_entropy(
                out.logits[0, logits_slice].float(),
                input_ids[0, labels_slice],
                reduction="sum",
            )
            nll += loss_sum.item()
            counted += scored
    del loaded
    torch.cuda.empty_cache()

    import math

    ppl = float(math.exp(nll / counted)) if counted else float("nan")
    return {
        "perplexity": ppl,
        "mean_nll": nll / counted if counted else None,
        "evaluated_tokens": counted,
        "windows": len(windows),
        "seq_len": seq_len,
        "stride": stride,
        "model_load_seconds": round(load_seconds, 3),
    }


def throughput(
    model: Path,
    *,
    prompt: str,
    max_tokens: int,
    dtype: str,
    seed: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model),
        dtype=dtype,
        seed=seed,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        enforce_eager=enforce_eager,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
    # Warmup (compiles/kernels) then timed.
    llm.generate([prompt], sampling)
    started = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - started
    if not outputs or not outputs[0].outputs:
        raise BenchError("vLLM generation returned empty outputs")
    completion = outputs[0].outputs[0]
    ntok = len(completion.token_ids)
    return {
        "prompt": prompt,
        "output_preview": completion.text[:200],
        "finish_reason": completion.finish_reason,
        "max_tokens": max_tokens,
        "generated_tokens": ntok,
        "decode_seconds": round(elapsed, 4),
        "tokens_per_second": round(ntok / elapsed, 2) if elapsed else None,
        "settings": {
            "dtype": dtype,
            "seed": seed,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 128,
            "enforce_eager": enforce_eager,
        },
    }


def run_bench(args: argparse.Namespace) -> None:
    """Measure one mode and write a partial receipt.

    Perplexity and throughput are separate processes on purpose. The BF16 arm
    alone is ~50 GiB of weights, so a vLLM engine started in the same process
    would find the GPU already occupied. Running each measurement in its own
    process guarantees both start from a clean GPU.
    """
    versions = package_versions()
    gpus = gpu_state()
    if args.require_gpu and not all(gpu["name"] == args.require_gpu for gpu in gpus):
        raise BenchError(
            f"required GPU is {args.require_gpu!r}, but live devices are "
            f"{[gpu['name'] for gpu in gpus]}"
        )
    identity = model_identity(args.model)
    result: dict[str, Any] = {
        "schema": "mlx-gptq-calibration/variant-bench/1",
        "recorded_at": recorded_at(),
        "label": args.label,
        "mode": args.mode,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runtime": versions,
        "gpus": gpus,
        "model": str(args.model),
        "identity": identity,
        "holdout": {
            "path": str(args.holdout),
            "sha256": sha256_file(args.holdout),
            "bytes": args.holdout.stat().st_size,
        },
    }
    if args.mode in ("perplexity", "all"):
        result["perplexity"] = perplexity(
            args.model,
            args.holdout,
            seq_len=args.seq_len,
            stride=args.stride,
            max_chunks=args.max_chunks,
            dtype=args.dtype,
        )
    if args.mode in ("throughput", "all"):
        result["throughput"] = throughput(
            args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            dtype=args.dtype,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
        )
    result["non_claims"] = [
        "Perplexity is held-out text NLL on the agentic-coding corpus; "
        "it is not a task or agentic score.",
        "Throughput is single-request decode under vLLM; it is not batched serving capacity.",
        "One variant, one GPU, one run; no cross-host or cross-protocol pooling.",
    ]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--mode", choices=("perplexity", "throughput", "all"), default="all")
    p.add_argument("--require-gpu")
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=64)
    p.add_argument("--prompt", default="Explain how a transformer attention head works.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_bench(args)
    except (ImportError, KeyError, OSError, ValueError) as exc:
        print(f"bench failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
