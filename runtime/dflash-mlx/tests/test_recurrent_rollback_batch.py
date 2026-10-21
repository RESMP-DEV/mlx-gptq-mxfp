from __future__ import annotations

import mlx.core as mx
import pytest
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache


def _cache(seed: int = 7) -> RecurrentRollbackCache:
    mx.random.seed(seed)
    batch, steps, heads, dv, dk, conv_dim = 3, 4, 2, 3, 8, 5
    cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
    cache[0] = mx.random.normal((batch, 3, conv_dim), key=mx.random.key(1))
    cache[1] = mx.random.normal((batch, heads, dv, dk), key=mx.random.key(2))
    cache.arm_rollback()
    cache.record_tape(
        tape=mx.random.normal((batch, steps, heads, dv), key=mx.random.key(3)),
        k=mx.random.normal((batch, steps, 1, dk), key=mx.random.key(4)),
        g=mx.random.normal((batch, steps, 1, dk), key=mx.random.key(5)),
        qkv=mx.random.normal((batch, steps, conv_dim), key=mx.random.key(6)),
    )
    return cache


def _serial_rollback(cache: RecurrentRollbackCache, row: int, accepted: int):
    result = RecurrentRollbackCache(
        len(cache.cache), conv_kernel_size=cache.conv_kernel_size
    )
    result[0] = cache._snapshot[0][row : row + 1]
    result[1] = cache._snapshot[1][row : row + 1]
    result.arm_rollback()
    result.record_tape(
        tape=cache._tape[row : row + 1],
        k=cache._tape_k[row : row + 1],
        g=cache._tape_g[row : row + 1],
        qkv=cache._tape_qkv[row : row + 1],
    )
    result.rollback(accepted)
    return result


def test_batch_rollback_matches_independent_rows() -> None:
    accepted = (0, 1, 3)
    batch = _cache()
    batch.rollback_batch(accepted)

    for row, count in enumerate(accepted):
        serial = _serial_rollback(_cache(), row, count)
        assert mx.array_equal(batch[0][row : row + 1], serial[0])
        assert mx.array_equal(batch[1][row : row + 1], serial[1])


def test_batch_rollback_validates_row_count_and_extent() -> None:
    cache = _cache()
    with pytest.raises(ValueError, match="batch mismatch"):
        cache.rollback_batch((0, 1))

    cache = _cache()
    with pytest.raises(ValueError, match="tape extent"):
        cache.rollback_batch((0, 1, 4))
