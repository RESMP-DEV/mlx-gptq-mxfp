#!/usr/bin/env python3
"""Run IFBench with Qwen3.8's upstream thinking and sampling contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from inspect_ai import eval
from inspect_ai.scorer import Scorer, Target, scorer
from inspect_ai.solver import TaskState
from openbench.config import load_task
from openbench.metrics.ifeval import ifeval_metrics
from openbench.monkeypatch.file_recorder_logfile_patch import (
    patch_file_recorder_logfile,
)
from openbench.scorers.ifbench import ifbench_scorer


def output_parts(state: TaskState) -> tuple[str, bool, int, str]:
    """Extract the visible answer and audit the runtime's reasoning split."""
    output = state.output
    reasoning_parts = []
    if output.choices:
        content = output.choices[0].message.content
        if isinstance(content, list):
            reasoning_parts = [
                part.reasoning
                for part in content
                if getattr(part, "type", None) == "reasoning"
            ]

    if reasoning_parts:
        reasoning = "\n".join(reasoning_parts)
        return (
            output.completion,
            bool(output.completion.strip()),
            len(reasoning),
            "structured",
        )

    reasoning, separator, answer = output.completion.rpartition("</think>")
    if separator:
        return answer.lstrip(), True, len(reasoning), "inline_fallback"
    return output.completion, False, 0, "none"


@scorer(metrics=[ifeval_metrics()])
def ifbench_final_answer_scorer() -> Scorer:
    """Score only the visible answer while leaving the raw trace in the log."""
    base_scorer = ifbench_scorer()

    async def score(state: TaskState, target: Target):
        raw_output = state.output
        answer, reasoning_closed, reasoning_chars, parser_mode = output_parts(state)
        if answer == raw_output.completion:
            result = await base_scorer(state, target)
        else:
            scoring_output = raw_output.model_copy(deep=True)
            scoring_output.completion = answer
            if scoring_output.choices:
                scoring_output.choices[0].message.text = answer
            state.output = scoring_output
            try:
                result = await base_scorer(state, target)
            finally:
                state.output = raw_output

        result.metadata = {
            **(result.metadata or {}),
            "reasoning_closed": reasoning_closed,
            "reasoning_chars": reasoning_chars,
            "final_answer_chars": len(answer),
            "raw_completion_chars": len(raw_output.completion),
            "reasoning_parser": parser_mode,
        }
        return result

    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--logfile", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_API_KEY", "local")
    patch_file_recorder_logfile(args.logfile)

    task = load_task("ifbench", allow_alpha=False)()
    task.scorer = [ifbench_final_answer_scorer()]

    logs = eval(
        tasks=[task],
        model=["vllm/default_model"],
        model_base_url=f"http://127.0.0.1:{args.port}/v1",
        epochs=1,
        limit=(args.start, args.end),
        fail_on_error=True,
        retry_on_error=1,
        log_samples=True,
        log_dir=str(log_dir),
        log_format="json",
        max_connections=1,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=0.0,
        extra_body={
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
                "preserve_thinking": True,
            },
        },
        max_tokens=args.max_tokens,
        seed=42,
        display="plain",
        timeout=7200,
        max_retries=1,
    )
    if not logs or logs[0].status != "success":
        raise SystemExit(f"evaluation failed: {logs}")


if __name__ == "__main__":
    main()
