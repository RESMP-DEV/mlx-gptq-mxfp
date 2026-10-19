import pytest
import torch

pytest.importorskip("triton")

from mlx_gptq.triton_dynamic_conv import triton_grouped_dynamic_convolve


def _reference(hidden, dynamic, base):
    output = torch.zeros_like(hidden)
    groups = dynamic.shape[-2]
    group_size = hidden.shape[-1] // groups
    grouped_output = output.view(*hidden.shape[:-1], groups, group_size)
    for index in range(hidden.shape[1]):
        for offset in range(base.shape[0]):
            if index >= offset:
                values = hidden[:, index - offset]
                grouped_base = base[offset].reshape(groups, group_size)
                grouped_output[:, index] += values.reshape(-1, groups, group_size) * (
                    grouped_base + dynamic[:, index, offset]
                )
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_triton_convolution_matches_eager_forward_and_gradients():
    torch.manual_seed(7)
    hidden = torch.randn(
        2, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dynamic = torch.randn(
        2, 5, 2, 4, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    base = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

    actual = triton_grouped_dynamic_convolve(hidden, dynamic, base, group_size=16)
    expected = _reference(hidden, dynamic, base)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    assert hidden.grad is not None
    assert dynamic.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(dynamic.grad).all()
