import numpy as np
import pytest

from mlx_gptq.packing import pack_mxfp4, unpack_q


def test_pack_mxfp4_is_lsb_first_and_byte_ordered():
    codes = np.arange(16, dtype=np.uint8)
    packed = pack_mxfp4(codes)
    assert packed.dtype == np.uint32
    assert packed.shape == (2,)
    assert unpack_q(packed, 4, 16).reshape(-1).tolist() == list(range(16))


def test_pack_mxfp4_rejects_unaligned_columns():
    with pytest.raises(AssertionError):
        pack_mxfp4(np.zeros(4, dtype=np.uint8))
