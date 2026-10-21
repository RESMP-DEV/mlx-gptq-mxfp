# Backend-neutral contracts shared by MLX serving and external engines.

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


class ServingContractError(ValueError):
    """Raised when a scheduler input violates a portable contract."""


def _candidate_better(
    candidate: tuple[float, int, int], root: tuple[float, int, int]
) -> bool:
    if candidate[0] != root[0]:
        return candidate[0] > root[0]
    if candidate[1] != root[1]:
        return candidate[1] < root[1]
    return candidate[2] < root[2]


@dataclass(frozen=True)
class ConfidenceVarlenPlan:
    """Contiguous slot-major rows for one variable-width MTP batch."""

    row_counts: tuple[int, ...]
    row_offsets: tuple[int, ...]
    allocated_rows: int


def plan_confidence_varlen(
    *,
    confidences: Sequence[float],
    proposal_offsets: Sequence[int],
    draft_maxima: Sequence[int],
    total_rows: int,
) -> ConfidenceVarlenPlan:
    """Allocate a global verify-row budget by cumulative draft confidence.

    Each slot contributes one golden row. Draft rows are ranked by the product
    of their conditional confidences, so selecting a later proposal also pays
    for every earlier proposal in that slot. Ties preserve slot-major input
    order. This is the Python contract for Strata's confidence-varlen policy.
    """

    slot_count = len(draft_maxima)
    if total_rows < 0:
        raise ServingContractError("total row budget is negative")
    if slot_count > total_rows:
        raise ServingContractError("budget is below one golden row per slot")
    if len(proposal_offsets) != slot_count + 1:
        raise ServingContractError("proposal offsets must have slot_count + 1 entries")
    if any(offset < 0 for offset in proposal_offsets):
        raise ServingContractError("proposal offsets are negative")
    if any(left > right for left, right in zip(proposal_offsets, proposal_offsets[1:])):
        raise ServingContractError("proposal offsets decrease")
    if any(maximum < 0 for maximum in draft_maxima):
        raise ServingContractError("draft maximum is negative")
    if proposal_offsets[0] != 0:
        raise ServingContractError("first proposal offset is not zero")
    if len(confidences) != proposal_offsets[-1]:
        raise ServingContractError("confidence count does not match proposal extents")

    extra_budget = total_rows - slot_count
    selected: list[tuple[float, int, int]] = []

    for slot in range(slot_count):
        start = proposal_offsets[slot]
        end = proposal_offsets[slot + 1]
        available = min(end - start, draft_maxima[slot], extra_budget)
        cumulative = 1.0
        for position in range(available):
            confidence = float(confidences[start + position])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ServingContractError("confidence outside [0, 1]")
            cumulative *= confidence
            if not cumulative > 0.0:
                break
            candidate = (cumulative, slot, position)
            heap_key = (cumulative, -slot, -position)
            if len(selected) < extra_budget:
                heapq.heappush(selected, heap_key)
            else:
                root = selected[0]
                if _candidate_better(candidate, (root[0], -root[1], -root[2])):
                    heapq.heapreplace(selected, heap_key)

    selected = [(item[0], -item[1], -item[2]) for item in selected]
    selected.sort(key=lambda item: (-item[0], item[1], item[2]))
    row_counts = [1] * slot_count
    for _, slot, position in selected:
        if position >= row_counts[slot]:
            raise ServingContractError("draft prefix invariant failed")
        row_counts[slot] += 1

    row_offsets: list[int] = []
    cursor = 0
    for count in row_counts:
        row_offsets.append(cursor)
        cursor += count
    return ConfidenceVarlenPlan(
        row_counts=tuple(row_counts),
        row_offsets=tuple(row_offsets),
        allocated_rows=cursor,
    )


@dataclass(frozen=True)
class AcceptancePlan:
    """Per-row acceptance and rollback groups after batched verification."""

    row_counts: tuple[int, ...]
    accepted_drafts: tuple[int, ...]

    @property
    def committed_rows(self) -> tuple[int, ...]:
        return tuple(value + 1 for value in self.accepted_drafts)

    def rollback_groups(self) -> dict[int, tuple[int, ...]]:
        groups: dict[int, list[int]] = {}
        for row, committed in enumerate(self.committed_rows):
            groups.setdefault(committed, []).append(row)
        return {count: tuple(rows) for count, rows in groups.items()}


def accepted_draft_lengths(
    *,
    proposed: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    row_counts: Sequence[int],
) -> AcceptancePlan:
    """Compare drafts with native greedy outputs and build rollback groups.

    ``proposed[i]`` excludes the golden token and contains the drafted tokens
    for row ``i``. ``target[i]`` contains the native greedy token at each
    verified position. The first mismatch rejects the remaining drafts.
    """

    if len(proposed) != len(target) or len(proposed) != len(row_counts):
        raise ServingContractError("acceptance row counts do not match")
    accepted: list[int] = []
    for row, count in enumerate(row_counts):
        drafts = proposed[row]
        posterior = target[row]
        if len(posterior) != count or len(drafts) != count - 1:
            raise ServingContractError("acceptance token counts do not match row plan")
        matched = 0
        for draft, native in zip(drafts, posterior[:-1]):
            if int(draft) != int(native):
                break
            matched += 1
        accepted.append(matched)
    return AcceptancePlan(
        row_counts=tuple(row_counts),
        accepted_drafts=tuple(accepted),
    )


@dataclass(frozen=True)
class WaitingRequest:
    """Minimal request metadata needed by backend-neutral admission."""

    request_id: str
    arrival_tick: int
    prompt_tokens: int
    prefix_hit_tokens: int = 0
    cancelled: bool = False

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_tokens - self.prefix_hit_tokens)


@dataclass(frozen=True)
class AdmissionPlan:
    """Short-first admission with bounded aging and cancellation."""

    admitted: tuple[str, ...]
    waiting: tuple[str, ...]
    cancelled: tuple[str, ...]


def plan_short_first_admission(
    *,
    requests: Sequence[WaitingRequest],
    now_tick: int,
    free_slots: int,
    aging_tokens_per_tick: float = 1.0,
) -> AdmissionPlan:
    """Pick short prefills first while preventing indefinite starvation."""

    if free_slots < 0:
        raise ServingContractError("free slot count is negative")
    if aging_tokens_per_tick < 0:
        raise ServingContractError("aging rate is negative")

    cancelled = tuple(req.request_id for req in requests if req.cancelled)
    candidates = [req for req in requests if not req.cancelled]
    for req in candidates:
        if req.prompt_tokens < 0 or not 0 <= req.prefix_hit_tokens <= req.prompt_tokens:
            raise ServingContractError("invalid prompt or prefix length")

    def priority(req: WaitingRequest) -> tuple[float, int, str]:
        age = max(0, now_tick - req.arrival_tick)
        aged_length = max(0.0, req.remaining_prefill - age * aging_tokens_per_tick)
        return (aged_length, req.arrival_tick, req.request_id)

    ordered = sorted(candidates, key=priority)
    admitted = tuple(req.request_id for req in ordered[: max(0, free_slots)])
    waiting = tuple(req.request_id for req in ordered[max(0, free_slots) :])
    return AdmissionPlan(admitted, waiting, cancelled)


def confidence_plan_to_json(plan: ConfidenceVarlenPlan) -> str:
    return json.dumps(asdict(plan), separators=(",", ":"), sort_keys=True)


def confidence_plan_from_json(value: str | dict[str, Any]) -> ConfidenceVarlenPlan:
    data = json.loads(value) if isinstance(value, str) else dict(value)
    try:
        return ConfidenceVarlenPlan(
            row_counts=tuple(int(item) for item in data["row_counts"]),
            row_offsets=tuple(int(item) for item in data["row_offsets"]),
            allocated_rows=int(data["allocated_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ServingContractError("invalid confidence plan JSON") from exc
