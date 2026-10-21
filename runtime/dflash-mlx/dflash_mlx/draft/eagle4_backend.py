# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

"""Correctness-first EAGLE-4 chain adapter for the DFlash engine."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.snapshot import TargetHiddenChunks
from dflash_mlx.engine.sampling import greedy_tokens_with_mask, masked_topk_arrays


class Eagle4DraftBackend:
    """Draft a greedy chain from the engine's projected target feature."""

    def make_cache(
        self,
        *,
        draft_model: Any,
        sink_size: int,
        window_size: int,
        allow_full_context_layers: bool = False,
    ) -> list[Any]:
        del draft_model, sink_size, window_size, allow_full_context_layers
        return []

    def _draft_chain_tokens(
        self,
        *,
        draft_model: Any,
        staged_first: mx.array,
        draft_context: Any,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
        top_width: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if int(block_len) <= 1:
            raise ValueError("EAGLE4 drafting requires block_len > 1")

        token_rows = [_root_token(staged_first)]
        root_feature = _last_feature(draft_context)
        position_offset = _root_position(draft_context)
        feature_rows = [root_feature[:, 0, :]]
        selected = mx.zeros((0,), dtype=mx.uint32)
        width = (
            0
            if top_width is None
            else min(int(top_width), int(draft_model.args.num_candidates))
        )
        top_ids = mx.zeros((0, width), dtype=mx.uint32)
        top_logprobs = mx.zeros((0, width), dtype=feature_rows[0].dtype)

        for _ in range(int(block_len) - 1):
            input_ids = mx.concatenate(token_rows)[None]
            projected = mx.concatenate(feature_rows, axis=0)[None]
            positions = mx.arange(
                position_offset,
                position_offset + len(token_rows),
                dtype=mx.int32,
            )[None]
            hidden = draft_model.forward_projected(input_ids, projected, positions)
            last_hidden = hidden[:, -1:, :]
            candidate_ids = draft_model.candidate_ids(last_hidden).reshape(-1)
            candidate_logits = draft_model.candidate_logits(last_hidden).reshape(-1)
            candidate_mask = (
                None
                if suppress_token_mask is None
                else mx.take(suppress_token_mask, candidate_ids, axis=0)
            )
            token_position = (
                greedy_tokens_with_mask(candidate_logits[None], candidate_mask)
                .reshape(1)
                .astype(mx.int32)
            )
            token = mx.take(candidate_ids, token_position).astype(mx.uint32)
            selected = mx.concatenate([selected, token], axis=0)

            if width:
                candidate_top, candidate_values = masked_topk_arrays(
                    candidate_logits[None], candidate_mask, width=width
                )
                candidate_top_ids = mx.take(
                    candidate_ids, candidate_top.astype(mx.int32)[0], axis=0
                )
                top_ids = mx.concatenate(
                    [top_ids, candidate_top_ids.astype(mx.uint32)[None]], axis=0
                )
                top_logprobs = mx.concatenate(
                    [
                        top_logprobs,
                        candidate_values.astype(top_logprobs.dtype)[0][None],
                    ],
                    axis=0,
                )
            feature_rows.append(hidden[:, -1:, :][:, 0, :])
            token_rows.append(token)

        return selected, top_ids, top_logprobs

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: Any,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: Any,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        del target_model, target_ops, draft_cache, mask_token_tail
        drafted, _top_ids, _top_values = self._draft_chain_tokens(
            draft_model=draft_model,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            suppress_token_mask=suppress_token_mask,
            top_width=None,
        )
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(drafted)
        return drafted

    def draft_greedy_capture(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: Any,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: Any,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        top_width: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        del target_model, target_ops, draft_cache, mask_token_tail
        drafted, top_ids, top_values = self._draft_chain_tokens(
            draft_model=draft_model,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            suppress_token_mask=suppress_token_mask,
            top_width=top_width,
        )
        if async_launch:
            mx.async_eval(drafted, top_ids, top_values)
        else:
            mx.eval(drafted, top_ids, top_values)
        return drafted, top_ids, top_values

    def propose_block(self, **_kwargs: Any) -> Any:
        raise ValueError("EAGLE4 backend supports greedy chain drafting only")

    def draft_with_topk(self, **_kwargs: Any) -> Any:
        raise ValueError("EAGLE4 backend does not support DDTree drafting")

    def draft_branch_blocks_batch(self, **_kwargs: Any) -> Any:
        raise ValueError("EAGLE4 backend does not support branch drafting")

    def advance_context(
        self,
        *,
        draft_model: Any,
        draft_cache: list[Any],
        draft_context: Any,
    ) -> None:
        del draft_model, draft_cache, draft_context


def _root_token(value: mx.array) -> mx.array:
    if (
        value.ndim != 1
        or value.shape[0] != 1
        or not mx.issubdtype(value.dtype, mx.integer)
    ):
        raise ValueError("EAGLE4 staged token must have integer shape [1]")
    return value.astype(mx.uint32).reshape(1)


def _last_feature(value: Any) -> mx.array:
    if isinstance(value, TargetHiddenChunks):
        if value.total_len < 1:
            raise ValueError("EAGLE4 target features cannot be empty")
        return value.slice(value.total_len - 1, value.total_len)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError("EAGLE4 target features must have shape [1, sequence, H]")
    return value[:, -1:, :]


def _root_position(value: Any) -> int:
    """Return the absolute position of the next target token."""
    if isinstance(value, TargetHiddenChunks):
        return int(value.total_len)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError("EAGLE4 target features must have shape [1, sequence, H]")
    return int(value.shape[1])
