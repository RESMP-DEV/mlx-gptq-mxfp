import pytest

mx = pytest.importorskip("mlx.core")

from mlx_gptq.awq_lfm import search_scale


def test_awq_search_never_regresses_unscaled_weighted_error():
    mx.random.seed(7)
    weight = mx.random.normal((24, 64)).astype(mx.bfloat16)
    second_moment = mx.exp(mx.linspace(-3, 3, 64))

    best_loss, alpha, _ = search_scale(weight, second_moment, grid=10)
    baseline_loss, baseline_alpha, _ = search_scale(weight, second_moment, grid=1)

    assert 0.0 <= alpha <= 1.0
    assert baseline_alpha in (0.0, 1.0)
    assert best_loss <= baseline_loss


def test_shortconv_scale_fusion_algebra():
    """Norm/input and C/output scaling preserve the dense block algebra."""
    mx.random.seed(11)
    hidden = 32
    x = mx.random.normal((5, hidden))
    norm = mx.random.uniform(shape=(hidden,), low=0.5, high=1.5)
    in_weight = mx.random.normal((3 * hidden, hidden))
    out_weight = mx.random.normal((hidden, hidden))
    in_scale = mx.random.uniform(shape=(hidden,), low=0.5, high=2.0)
    out_scale = mx.random.uniform(shape=(hidden,), low=0.5, high=2.0)

    projected = (x * norm) @ in_weight.T
    b, c, value = mx.split(projected, 3, axis=-1)
    reference = (c * (b * value)) @ out_weight.T

    transformed = mx.concatenate(
        [
            in_weight[:hidden],
            in_weight[hidden : 2 * hidden] / out_scale[:, None],
            in_weight[2 * hidden :],
        ],
        axis=0,
    )
    transformed = transformed * in_scale[None, :]
    projected = (x * (norm / in_scale)) @ transformed.T
    b, c, value = mx.split(projected, 3, axis=-1)
    actual = (c * (b * value)) @ (out_weight * out_scale[None, :]).T
    mx.eval(reference, actual)

    assert mx.allclose(reference, actual, rtol=2e-5, atol=2e-5).item()
