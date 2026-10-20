#!/usr/bin/env python3
"""Launch mlx-lm with CUDA-specific distributed and sampling fixes."""

import mlx.core as mx
import mlx_lm.server as server
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

_original_sharded_load = server.sharded_load
_original_make_sampler = server._make_sampler


def _cuda_safe_sharded_load(*args, **kwargs):
    original_all_sum = mx.distributed.all_sum

    def cuda_safe_all_sum(value, *all_sum_args, **all_sum_kwargs):
        if all_sum_kwargs.get("stream") == mx.cpu:
            all_sum_kwargs["stream"] = None
        return original_all_sum(value, *all_sum_args, **all_sum_kwargs)

    mx.distributed.all_sum = cuda_safe_all_sum
    try:
        return _original_sharded_load(*args, **kwargs)
    finally:
        mx.distributed.all_sum = original_all_sum


server.sharded_load = _cuda_safe_sharded_load


def _cuda_safe_make_sampler(args, tokenizer):
    """Use explicit PRNG keys because CUDA worker-thread RNG is degenerate."""
    sampling = args.sampling
    if (
        sampling.temperature == 0
        or args.seed is None
        or mx.metal.is_available()
        or sampling.xtc_probability > 0
    ):
        return _original_make_sampler(args, tokenizer)

    key = mx.random.key(args.seed)

    def sampler(logprobs):
        nonlocal key
        if 0 < sampling.top_p < 1.0:
            logprobs = apply_top_p(logprobs, sampling.top_p)
        if sampling.min_p != 0.0:
            logprobs = apply_min_p(logprobs, sampling.min_p)
        if sampling.top_k > 0:
            logprobs = apply_top_k(logprobs, sampling.top_k)
        key, sample_key = mx.random.split(key)
        return mx.random.categorical(
            logprobs * (1 / sampling.temperature), key=sample_key
        )

    return sampler


server._make_sampler = _cuda_safe_make_sampler


if __name__ == "__main__":
    server.main()
