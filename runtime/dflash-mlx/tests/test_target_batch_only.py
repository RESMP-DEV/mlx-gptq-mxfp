from __future__ import annotations

from types import SimpleNamespace

import pytest
from dflash_mlx.server import model_provider
from dflash_mlx.server.config import build_parser, normalize_cli_args


class _MergeableCache:
    def merge(self, _caches):
        return self


class _PlainCache:
    pass


class _TargetModel:
    def __init__(self, mergeable: bool = True):
        self.mergeable = mergeable

    def parameters(self):
        return []

    def make_cache(self):
        return [_MergeableCache() if self.mergeable else _PlainCache()]


def _provider_args(*, mergeable: bool = True):
    args = build_parser().parse_args(["--model", "target", "--target-batch-only"])
    normalize_cli_args(args)
    return args, _TargetModel(mergeable=mergeable)


@pytest.mark.parametrize("mergeable", [True, False])
def test_target_batch_only_loads_without_drafter(monkeypatch, mergeable: bool):
    args, target_model = _provider_args(mergeable=mergeable)
    target_ops = object()
    calls = []

    class FakeGroup:
        def size(self):
            return 1

    monkeypatch.setattr(model_provider.mx.distributed, "init", lambda: FakeGroup())
    monkeypatch.setattr(
        model_provider,
        "load_target_bundle",
        lambda *_args, **_kwargs: (
            calls.append((_args, _kwargs))
            or SimpleNamespace(
                model=target_model,
                tokenizer=SimpleNamespace(
                    chat_template=None, default_chat_template=None
                ),
                meta={"resolved_model_ref": "target"},
                target_ops=target_ops,
            )
        ),
    )

    provider = model_provider.DFlashModelProvider(args)
    provider.load("default_model")

    assert calls
    assert calls[0][1]["install_speculative_hooks"] is False
    assert provider.draft_model is None
    assert provider.draft_backend is None
    assert provider.target_ops is target_ops
    assert provider.is_batchable is mergeable
    assert provider.model_key == ("target", None, None)
    assert model_provider._runtime_components_ready(provider) is True


def test_target_batch_only_rejects_explicit_drafter():
    args = build_parser().parse_args(
        ["--model", "target", "--target-batch-only", "--draft", "drafter"]
    )
    with pytest.raises(SystemExit, match="--target-batch-only"):
        normalize_cli_args(args)


def test_regular_mode_still_requires_drafter_for_readiness():
    provider = SimpleNamespace(
        model_key=("target", None, "draft"),
        target_ops=object(),
        draft_backend=None,
        cli_args=SimpleNamespace(target_batch_only=False),
    )
    assert model_provider._runtime_components_ready(provider) is False
