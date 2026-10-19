import torch
from pathlib import Path
from safetensors.torch import save_file

from tools.merge_dflash2_candidate import SELECTOR, merge


def _tiny_base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    weights = {
        "fc.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        SELECTOR: torch.zeros(2, 4, dtype=torch.bfloat16),
    }
    save_file(weights, str(base / "model.safetensors"), metadata=None)
    (base / "config.json").write_text('{"model_type":"qwen3"}')
    return base


def _tiny_candidate(tmp_path: Path, a: torch.Tensor, b: torch.Tensor, proj: torch.Tensor) -> Path:
    cand = tmp_path / "run"
    cand.mkdir(parents=True)
    save_file(
        {
            "fc.a": a,
            "fc.b": b,
            "fc.scaling": torch.tensor([1.0]),
        },
        str(cand / "lora_adapters.safetensors"),
    )
    save_file({"weight": proj}, str(cand / "distilled_hidden_projection.safetensors"))
    return cand


def test_merge_applies_delta_and_selector_lerp(tmp_path):
    torch.manual_seed(5)
    base = _tiny_base(tmp_path)
    a = torch.randn(2, 8)
    b = torch.randn(4, 2)
    proj = torch.randn(2, 4).to(torch.bfloat16)
    cand = _tiny_candidate(tmp_path, a, b, proj)

    out = tmp_path / "merged"
    receipt = merge(base, cand, out, strength=1.0)
    from safetensors.torch import load_file

    weights = load_file(str(out / "model.safetensors"), device="cpu")
    expected_delta = ((b.float() @ a.float()) * 1.0).to(torch.bfloat16)
    torch.testing.assert_close(weights["fc.weight"], expected_delta)
    torch.testing.assert_close(weights[SELECTOR], proj)
    assert receipt["adapter_prefixes"] == 1
    assert (out / "config.json").is_file()
    assert (out / "blend.json").is_file()


def test_merge_strength_half_scales_delta_and_selector(tmp_path):
    base = _tiny_base(tmp_path)
    a = torch.randn(2, 8)
    b = torch.randn(4, 2)
    proj = torch.randn(2, 4).to(torch.bfloat16)
    cand = _tiny_candidate(tmp_path / "c2", a, b, proj)

    out = tmp_path / "merged_half"
    merge(base, cand, out, strength=0.5)
    from safetensors.torch import load_file

    weights = load_file(str(out / "model.safetensors"), device="cpu")
    expected_delta = ((b.float() @ a.float()) * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(weights["fc.weight"], expected_delta)
    expected_selector = (proj.float() * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(weights[SELECTOR], expected_selector)
