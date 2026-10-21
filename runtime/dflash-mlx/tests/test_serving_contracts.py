from __future__ import annotations

import math
import random

import pytest
from dflash_mlx.server.serving_contracts import (
    AcceptancePlan,
    ConfidenceVarlenPlan,
    ServingContractError,
    WaitingRequest,
    accepted_draft_lengths,
    confidence_plan_from_json,
    confidence_plan_to_json,
    plan_confidence_varlen,
    plan_short_first_admission,
)


def test_confidence_plan_uses_cumulative_products() -> None:
    plan = plan_confidence_varlen(
        confidences=[1.0, 0.5, 0.25, 0.8, 0.8, 0.6],
        proposal_offsets=[0, 3, 5, 6],
        draft_maxima=[3, 2, 1],
        total_rows=7,
    )

    assert plan == ConfidenceVarlenPlan(
        row_counts=(2, 3, 2),
        row_offsets=(0, 2, 5),
        allocated_rows=7,
    )


def test_confidence_plan_respects_caps_and_zero_confidence() -> None:
    capped = plan_confidence_varlen(
        confidences=[0.9, 0.9, 1.0, 1.0],
        proposal_offsets=[0, 2, 4],
        draft_maxima=[0, 2],
        total_rows=4,
    )
    assert capped.row_counts == (1, 3)

    zero = plan_confidence_varlen(
        confidences=[0.0, 0.9, 0.5, 0.0],
        proposal_offsets=[0, 2, 4],
        draft_maxima=[2, 2],
        total_rows=8,
    )
    assert zero.row_counts == (1, 2)


def test_confidence_plan_keeps_slot_major_ties() -> None:
    plan = plan_confidence_varlen(
        confidences=[1.0, 1.0, 1.0, 1.0],
        proposal_offsets=[0, 2, 4],
        draft_maxima=[2, 2],
        total_rows=3,
    )
    assert plan.row_counts == (2, 1)


def test_confidence_plan_rejects_invalid_inputs() -> None:
    for confidences, offsets, maxima, total in (
        ([1.5], [0, 1], [1], 2),
        ([math.nan], [0, 1], [1], 2),
        ([1.0], [1, 2], [1], 2),
        ([1.0, 1.5], [0, 2], [2, 1], 3),
    ):
        with pytest.raises(ServingContractError):
            plan_confidence_varlen(
                confidences=confidences,
                proposal_offsets=offsets,
                draft_maxima=maxima,
                total_rows=total,
            )


def test_confidence_plan_matches_bruteforce_ranking() -> None:
    rng = random.Random(20261021)
    for _case in range(100):
        slot_count = rng.randint(1, 4)
        lengths = [rng.randint(0, 4) for _ in range(slot_count)]
        maxima = [rng.randint(0, length) for length in lengths]
        values = [rng.random() for _ in range(sum(lengths))]
        total_rows = slot_count + rng.randint(0, sum(maxima))

        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        plan = plan_confidence_varlen(
            confidences=values,
            proposal_offsets=offsets,
            draft_maxima=maxima,
            total_rows=total_rows,
        )

        ranked = []
        cursor = 0
        for slot, (length, maximum) in enumerate(zip(lengths, maxima)):
            cumulative = 1.0
            for position in range(min(length, maximum)):
                cumulative *= values[cursor + position]
                if cumulative <= 0.0:
                    break
                ranked.append((cumulative, slot, position))
            cursor += length
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        expected = [1] * slot_count
        for _, slot, position in ranked[: total_rows - slot_count]:
            expected[slot] += 1

        assert plan.row_counts == tuple(expected)
        assert plan.allocated_rows == sum(expected)


def test_acceptance_groups_rows_by_rollback_length() -> None:
    result = accepted_draft_lengths(
        proposed=[[7, 8], [7, 9], [7, 8, 10]],
        target=[[7, 8, 20], [7, 20, 21], [7, 8, 10, 11]],
        row_counts=[3, 3, 4],
    )

    assert result == AcceptancePlan(
        row_counts=(3, 3, 4),
        accepted_drafts=(2, 1, 3),
    )
    assert result.committed_rows == (3, 2, 4)
    assert result.rollback_groups() == {3: (0,), 2: (1,), 4: (2,)}


def test_acceptance_rejects_shape_mismatch() -> None:
    with pytest.raises(ServingContractError):
        accepted_draft_lengths(
            proposed=[[1]],
            target=[[1, 2, 3]],
            row_counts=[2],
        )


def test_short_first_admission_ages_long_requests() -> None:
    requests = [
        WaitingRequest("long", arrival_tick=0, prompt_tokens=100),
        WaitingRequest("short", arrival_tick=20, prompt_tokens=10),
    ]

    early = plan_short_first_admission(
        requests=requests,
        now_tick=30,
        free_slots=1,
        aging_tokens_per_tick=1.0,
    )
    assert early.admitted == ("short",)

    aged = plan_short_first_admission(
        requests=requests,
        now_tick=110,
        free_slots=1,
        aging_tokens_per_tick=1.0,
    )
    assert aged.admitted == ("long",)


def test_short_first_admission_separates_cancelled_requests() -> None:
    requests = [
        WaitingRequest("a", arrival_tick=0, prompt_tokens=4),
        WaitingRequest("b", arrival_tick=1, prompt_tokens=8, cancelled=True),
    ]
    plan = plan_short_first_admission(
        requests=requests,
        now_tick=2,
        free_slots=1,
    )

    assert plan.admitted == ("a",)
    assert plan.waiting == ()
    assert plan.cancelled == ("b",)


def test_confidence_plan_json_round_trip() -> None:
    plan = ConfidenceVarlenPlan(
        row_counts=(2, 1),
        row_offsets=(0, 2),
        allocated_rows=3,
    )
    assert confidence_plan_from_json(confidence_plan_to_json(plan)) == plan

    with pytest.raises(ServingContractError):
        confidence_plan_from_json({"row_counts": [2]})
