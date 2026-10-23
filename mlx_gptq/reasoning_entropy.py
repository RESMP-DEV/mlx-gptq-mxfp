"""Top-logprob entropy calibration and cut/continue policy for Flint."""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Completion:
    content: str
    tokens: int
    entropy: float | None
    wall_seconds: float
    finish_reason: str | None


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    expected: str
    probe_suffix: str


def load_tasks(path: Path, limit: int, offset: int = 0) -> list[Task]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            expected = row.get("expected")
            if not isinstance(prompt, str) or not isinstance(expected, str):
                raise ValueError(
                    f"{path}:{line_number}: prompt and expected must be strings"
                )
            tasks.append(
                Task(
                    task_id=str(row.get("id", line_number - 1)),
                    prompt=prompt,
                    expected=expected,
                    probe_suffix=str(
                        row.get("probe_suffix", "\n</think>\n\nFinal Answer:")
                    ),
                )
            )
    selected = tasks[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"{path}: requested {limit} tasks after offset {offset}, "
            f"found {len(selected)}"
        )
    return selected


def initial_prompt(task: Task, assistant_prefix: str) -> str:
    return task.prompt + assistant_prefix


def probe_prompt(task: Task, prefix: str, assistant_prefix: str) -> str:
    return initial_prompt(task, assistant_prefix) + prefix + task.probe_suffix


def make_case(
    task: Task,
    answer: str,
    wall_seconds: float,
    tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "hit": answer_matches(answer, task.expected),
        "wall_seconds": wall_seconds,
        "completion_tokens": tokens,
        **extra,
    }


def calibrate(
    base_url: str,
    model: str,
    tasks: Sequence[Task],
    assistant_prefix: str,
    checkpoint_tokens: int,
    probe_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    for task in tasks:
        prefix = complete(
            base_url,
            model,
            initial_prompt(task, assistant_prefix),
            checkpoint_tokens,
            timeout,
            top_logprobs=20,
        )
        probe = complete(
            base_url,
            model,
            probe_prompt(task, prefix.content, assistant_prefix),
            probe_tokens,
            timeout,
        )
        cases.append(
            make_case(
                task,
                probe.content,
                prefix.wall_seconds + probe.wall_seconds,
                prefix.tokens + probe.tokens,
                checkpoint_entropy=prefix.entropy,
                reasoning_prefix=prefix.content,
                forced_answer=probe.content,
            )
        )
    correct = [float(row["checkpoint_entropy"]) for row in cases if row["hit"]]
    incorrect = [float(row["checkpoint_entropy"]) for row in cases if not row["hit"]]
    values = thresholds(correct, incorrect)
    return cases, {
        **values,
        "incorrect_min_entropy": min(incorrect) if incorrect else None,
    }


def evaluate(
    base_url: str,
    model: str,
    tasks: Sequence[Task],
    assistant_prefix: str,
    checkpoint_tokens: int,
    probe_tokens: int,
    max_tokens: int,
    timeout: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baselines = []
    policies = []
    for task in tasks:
        prompt = initial_prompt(task, assistant_prefix)
        base = complete(
            base_url,
            model,
            prompt,
            max_tokens,
            timeout,
            top_logprobs=20,
        )
        baselines.append(
            make_case(
                task,
                final_answer(base.content),
                base.wall_seconds,
                base.tokens,
                sequence_entropy=base.entropy,
                finish_reason=base.finish_reason,
                raw_content=base.content,
            )
        )
        prefix = complete(
            base_url,
            model,
            prompt,
            checkpoint_tokens,
            timeout,
            top_logprobs=20,
        )
        action = decide(prefix.entropy, threshold)
        if action == "cut":
            probe = complete(
                base_url,
                model,
                probe_prompt(task, prefix.content, assistant_prefix),
                probe_tokens,
                timeout,
            )
            policies.append(
                make_case(
                    task,
                    probe.content,
                    prefix.wall_seconds + probe.wall_seconds,
                    prefix.tokens + probe.tokens,
                    decision=action,
                    checkpoint_entropy=prefix.entropy,
                    reasoning_prefix=prefix.content,
                )
            )
        else:
            continued = complete(
                base_url,
                model,
                prompt + prefix.content,
                max_tokens - checkpoint_tokens,
                timeout,
            )
            policies.append(
                make_case(
                    task,
                    final_answer(prefix.content + continued.content),
                    prefix.wall_seconds + continued.wall_seconds,
                    prefix.tokens + continued.tokens,
                    decision=action,
                    checkpoint_entropy=prefix.entropy,
                    reasoning_prefix=prefix.content,
                    raw_content=prefix.content + continued.content,
                )
            )
    return baselines, policies


def entropy(top_logprobs: Sequence[Mapping[str, Any]]) -> float:
    """Return normalized Shannon entropy in bits from token logprobs."""
    values = [float(item["logprob"]) for item in top_logprobs]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("invalid top_logprobs")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(probability * math.log2(probability) for probability in probabilities)


def mean_entropy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [entropy(row["top_logprobs"]) for row in rows if row.get("top_logprobs")]
    return statistics.fmean(values) if values else None


def thresholds(
    correct: Sequence[float], incorrect: Sequence[float]
) -> dict[str, float]:
    if not correct:
        raise ValueError("no correct calibration probes")
    return {
        "mean_correct": statistics.fmean(correct),
        "zero_observed_error": min(incorrect) - 1e-9
        if incorrect
        else max(correct) + 1e-9,
    }


def decide(entropy_value: float | None, threshold: float) -> str:
    """Missing entropy always continues; it never cuts blindly."""
    if entropy_value is not None and entropy_value <= threshold:
        return "cut"
    return "continue"


def answer_matches(answer: str, expected: str) -> bool:
    return expected.casefold() in answer.casefold()


def final_answer(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    walls = [float(item["wall_seconds"]) for item in cases]
    tokens = [int(item["completion_tokens"]) for item in cases]
    return {
        "attempted": len(cases),
        "correct": sum(bool(item["hit"]) for item in cases),
        "mean_tokens": statistics.fmean(tokens) if tokens else None,
        "median_tokens": statistics.median(tokens) if tokens else None,
        "mean_wall_seconds": statistics.fmean(walls) if walls else None,
        "median_wall_seconds": percentile(walls, 0.5),
        "p95_wall_seconds": percentile(walls, 0.95),
    }


def complete(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    top_logprobs: int | None = None,
) -> Completion:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if top_logprobs is not None:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

    choice = result.get("choices", [{}])[0]
    logprob_payload = choice.get("logprobs", {})
    rows = logprob_payload.get("content", logprob_payload.get("tokens", []))
    usage = result.get("usage", {})
    return Completion(
        content=str(choice.get("text", "")),
        tokens=int(usage.get("completion_tokens", len(rows))),
        entropy=mean_entropy(rows),
        wall_seconds=time.perf_counter() - started,
        finish_reason=choice.get("finish_reason"),
    )
