"""PyTorch implementation of the Qwen4Exp one-layer native MTP drafter.

Transformers 5.18 executes the Qwen4Exp trunk but deliberately ignores the
checkpoint's ``mtp.*`` tensors.  This module reuses the trunk's exact attention,
gated-residual, and sparse-MoE blocks, then supplies only the MTP-specific input
fusion and output mixer transcribed from Strata's live native runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpTextSparseMoeBlock,
)

from .qat import fake_quantize_mxfp4


class FlashNextMTP(torch.nn.Module):
    """One native Qwen4Exp MTP layer plus the frozen target embedding/head."""

    def __init__(self, config: Any, attention_implementation: str = "sdpa"):
        super().__init__()
        if attention_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                "FlashNextMTP supports only eager or SDPA attention, got "
                f"{attention_implementation!r}"
            )
        text = config.text_config
        text._attn_implementation = attention_implementation
        self.config = text
        self.fc_embedding = torch.nn.Linear(
            text.hidden_size, text.hidden_size, bias=False
        )
        self.fc_hidden = torch.nn.Linear(text.hidden_size, text.hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(
            text.hidden_size, eps=text.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
            text.hc_count * text.hidden_size, eps=text.rms_norm_eps
        )
        self.self_attn = Qwen4ExpTextAttention(text, 0)
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=text)
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(text)
        self.mlp = Qwen4ExpTextSparseMoeBlock(text)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(text)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(text, use_combine=False)
        self.token_embedding = torch.nn.Embedding(text.vocab_size, text.hidden_size)
        self.lm_head = torch.nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.qat_mxfp4 = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
        attention_implementation: str = "sdpa",
    ) -> "FlashNextMTP":
        """Load the 31 mtp tensors and shared target embedding/head exactly."""
        checkpoint = Path(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        model = cls(config, attention_implementation=attention_implementation).to(
            device=device, dtype=dtype
        )
        index = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]
        groups: dict[str, list[str]] = {}
        for source_name, shard in index.items():
            target_name = None
            if source_name.startswith("mtp.layers.0."):
                target_name = source_name.removeprefix("mtp.layers.0.")
            elif source_name.startswith("mtp."):
                target_name = source_name.removeprefix("mtp.")

            elif source_name == "model.language_model.embed_tokens.weight":
                target_name = "token_embedding.weight"
            elif source_name == "lm_head.weight":
                target_name = "lm_head.weight"
            if target_name is not None:
                groups.setdefault(shard, []).append((source_name, target_name))

        parameters = dict(model.named_parameters())
        loaded: set[str] = set()
        for shard, names in sorted(groups.items()):
            tensors = load_file(str(checkpoint / shard), device=str(device))
            for source_name, target_name in names:
                value = tensors[source_name].to(dtype=dtype)
                expected = parameters[target_name].shape
                if tuple(value.shape) != tuple(expected):
                    raise ValueError(
                        f"{source_name} shape {tuple(value.shape)} does not match "
                        f"{target_name} {tuple(expected)}"
                    )
                with torch.no_grad():
                    parameters[target_name].copy_(value)
                loaded.add(target_name)
                del value
            del tensors

        expected = set(parameters)
        missing = sorted(expected - loaded)
        if missing:
            raise KeyError(f"Flash-Next MTP tensors not loaded: {missing[:8]}")
        model.eval()
        model.requires_grad_(False)
        return model

    def _effective_weight(self, module: torch.nn.Linear) -> torch.Tensor:
        weight = module.weight
        return fake_quantize_mxfp4(weight) if self.qat_mxfp4 else weight

    def _linear(self, input: torch.Tensor, module: torch.nn.Linear) -> torch.Tensor:
        return torch.nn.functional.linear(
            input, self._effective_weight(module).to(input.dtype)
        )

    def _project(self, input: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        if hasattr(module, "effective_weight"):
            return module(input)
        return self._linear(input, cast(torch.nn.Linear, module))

    def forward(
        self,
        residual: torch.Tensor,
        input_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return two-ahead draft logits from target residual rows.

        ``residual`` is the verifier's stacked final residual ``[B, T, 4*H]``.
        ``input_tokens`` is the token at each source position plus one.  The
        native cell attends densely over its own cells; in this training forward
        a batch is an independent MTP sequence unless positions and a cache say
        otherwise.
        """
        if (
            residual.ndim != 3
            or residual.shape[-1] != self.config.hc_count * self.config.hidden_size
        ):
            raise ValueError("residual must be [batch, rows, 10240] for Flash-Next MTP")
        if input_tokens.shape != residual.shape[:2]:
            raise ValueError("input token shape must match residual rows")
        batch, rows, _ = residual.shape
        if positions is None:
            positions = torch.arange(rows, device=residual.device).repeat(batch, 1)
        elif tuple(positions.shape) != (batch, rows):
            raise ValueError("positions must have the same shape as input tokens")

        dtype = self.pre_fc_norm_embedding.weight.dtype
        residual = residual.to(dtype)
        embedding = self.token_embedding(input_tokens)
        embedding = self.pre_fc_norm_embedding(embedding)
        embedding_projection = self._project(embedding, self.fc_embedding)

        streams = residual.unflatten(
            -1, (self.config.hc_count, self.config.hidden_size)
        )
        normalized_streams = self.pre_fc_norm_hidden(residual)
        normalized_streams = normalized_streams.unflatten(
            -1, (self.config.hc_count, self.config.hidden_size)
        )
        projected_streams = self._project(
            normalized_streams.reshape(-1, self.config.hidden_size), self.fc_hidden
        ).reshape(streams.shape)
        streams = projected_streams + embedding_projection.unsqueeze(-2)
        hyper = streams.flatten(-2)

        mixed, hyper, injection = self.attn_hyper_connection(hyper)
        attn, _ = self.self_attn(
            mixed,
            self._position_embeddings(mixed, positions),
            torch.ones(
                (batch, 1, rows, rows),
                dtype=torch.bool,
                device=mixed.device,
            ).tril(),
            None,
        )
        injection = attn.unsqueeze(-2) * injection.unsqueeze(-1)
        hyper = hyper + injection.flatten(-2)

        mixed, hyper, injection = self.mlp_hyper_connection(hyper)
        mlp = self.mlp(mixed)
        injection = mlp.unsqueeze(-2) * injection.unsqueeze(-1)
        hyper = hyper + injection.flatten(-2)

        sample = self.hyper_connection_mixer(hyper)
        return self._linear(sample, self.lm_head)

    def _position_embeddings(
        self, sample: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Text uses the same position in all three MRoPE sections here.
        return self.rotary_emb(sample, positions.unsqueeze(0).expand(3, -1, -1))
