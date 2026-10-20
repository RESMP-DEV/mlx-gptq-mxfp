"""Exact DFlash2 MXFP4 projection shapes for kernel qualification.

The shapes come from the promoted generation-3 draft head, not from a
generic MoE model. A kernel is qualified on these shapes or it is not
qualified for DFlash2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearShape:
    """One DFlash2 linear projection."""

    name: str
    rows: int
    columns: int

    @property
    def input_features(self) -> int:
        return self.columns

    @property
    def output_features(self) -> int:
        return self.rows


# Unique projections of the 5-layer DFlash2 draft head.
LINEARS: tuple[LinearShape, ...] = (
    LinearShape("fc", 5120, 25600),
    LinearShape("attention_conv.kernel_projection", 1280, 5120),
    LinearShape("mlp.down_proj", 5120, 17408),
    LinearShape("mlp.gate_proj", 17408, 5120),
    LinearShape("mlp.up_proj", 17408, 5120),
    LinearShape("mlp_conv.kernel_projection", 1280, 5120),
    LinearShape("self_attn.k_proj", 1024, 5120),
    LinearShape("self_attn.o_proj", 5120, 4096),
    LinearShape("self_attn.q_proj", 4096, 5120),
    LinearShape("self_attn.v_proj", 1024, 5120),
)

# Row counts that matter: training prefill batches and decode batches.
ROW_COUNTS: tuple[int, ...] = (1, 5, 16, 128, 512, 2048, 8192)


def qualification_shapes() -> list[tuple[str, int, int, int]]:
    """Return (name, rows, output_features, input_features) tuples."""
    return [
        (shape.name, rows, shape.output_features, shape.input_features)
        for shape in LINEARS
        for rows in ROW_COUNTS
    ]


def dominant_shapes() -> list[tuple[str, int, int, int]]:
    """Return the shapes that dominate DFlash2 training and serving time."""
    selected = {"fc", "mlp.gate_proj", "mlp.down_proj", "self_attn.q_proj"}
    return [
        (shape.name, rows, shape.output_features, shape.input_features)
        for shape in LINEARS
        if shape.name in selected
        for rows in (5, 128, 2048, 8192)
    ]
