"""GPTQ solver: output-space error must beat RTN, groups must be honored."""

import pytest
import torch

from mlx_gptq.gptq import solve_gptq
from mlx_gptq.grid import dequantize_artifact, rtn_quantize, rtn_quantize_mxfp


def _setup(B=2, R=32, C=128, N=512, seed=0):
    torch.manual_seed(seed)
    # correlated inputs make error compensation matter
    base = torch.randn(N, C)
    X = base + 0.5 * torch.roll(base, 1, dims=1)
    W = torch.randn(B, R, C)
    H = (X.T @ X).unsqueeze(0).expand(B, C, C).contiguous()
    return W, H, X


def test_gptq_beats_rtn_on_output_error():
    W, H, X = _setup()
    q, s, b, W_dq, rel = solve_gptq(W, H, bits=4, group_size=32)
    _, _, _, W_rtn = rtn_quantize(W, bits=4, group_size=32)
    err_gptq = ((X @ (W_dq - W).transpose(1, 2)) ** 2).mean()
    err_rtn = ((X @ (W_rtn - W).transpose(1, 2)) ** 2).mean()
    assert err_gptq < err_rtn * 0.9, (err_gptq, err_rtn)
    assert rel.shape == (2,) and (rel > 0).all() and (rel < 0.5).all()


def test_gptq_dequant_consistency():
    """W_dq must equal exactly q*scale+bias with the emitted params."""
    W, H, _ = _setup(B=1, R=8, C=64)
    q, s, b, W_dq, _ = solve_gptq(W, H, bits=4, group_size=32,
                                  storage_dtype=torch.float32)
    G = 32
    manual = (q.float().reshape(1, 8, -1, G) * s.float().unsqueeze(-1)
              + b.float().unsqueeze(-1)).reshape(1, 8, 64)
    assert torch.allclose(manual, W_dq, atol=1e-5)
    assert int(q.max()) <= 15


def test_gptq_batch_independent():
    """Batched solve must equal per-item solves."""
    W, H, _ = _setup(B=2)
    H1 = H.clone(); H1[1] *= 3.0  # make items differ
    qb, sb, bb, _, _ = solve_gptq(W, H1, bits=4, group_size=32)
    q0, s0, b0, _, _ = solve_gptq(W[:1], H1[:1], bits=4, group_size=32)
    assert torch.equal(qb[0], q0[0])
    assert torch.equal(sb[0], s0[0])


def test_gptq_zero_hessian_falls_back():
    """All-dead Hessian (unrouted expert) must not crash and stays finite."""
    W = torch.randn(1, 8, 64)
    H = torch.zeros(1, 64, 64)
    q, s, b, W_dq, rel = solve_gptq(W, H, bits=4, group_size=32)
    assert torch.isfinite(W_dq).all()


@torch.no_grad()
@pytest.mark.parametrize("mode,bits", [("mxfp4", 4), ("mxfp8", 8)])
def test_mxfp_gptq_dequant_consistency(mode, bits):
    W, H, _ = _setup(B=1, R=8, C=64)
    q, s, b, W_dq, rel = solve_gptq(
        W, H, bits=bits, group_size=32, mode=mode
    )
    manual = dequantize_artifact(q, s, b, mode, 32)
    assert b is None and s.dtype == torch.uint8
    assert torch.equal(manual, W_dq)
    assert torch.isfinite(W_dq).all()
    assert (rel > 0).all()


@torch.no_grad()
@pytest.mark.parametrize("mode,bits", [("mxfp4", 4), ("mxfp8", 8)])
def test_mxfp_gptq_improves_activation_error(mode, bits):
    W, H, X = _setup(B=1, R=32, C=128, N=1024, seed=11)
    _, _, _, W_gptq, _ = solve_gptq(
        W, H, bits=bits, group_size=32, mode=mode
    )
    _, _, _, W_rtn = rtn_quantize_mxfp(W, mode)
    err_gptq = ((X @ (W_gptq - W).transpose(1, 2)) ** 2).mean()
    err_rtn = ((X @ (W_rtn - W).transpose(1, 2)) ** 2).mean()
    assert err_gptq < err_rtn, (mode, err_gptq, err_rtn)
