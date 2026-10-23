from __future__ import annotations

import json
import math

from mlx_gptq.reasoning_entropy import (
    complete,
    decide,
    entropy,
    final_answer,
    thresholds,
)


def test_entropy_normalizes_top_logprobs() -> None:
    value = entropy([{"logprob": math.log(0.75)}, {"logprob": math.log(0.25)}])
    assert math.isclose(value, 0.8112781244591328)


def test_entropy_rejects_invalid_payload() -> None:
    try:
        entropy([])
    except ValueError as exc:
        assert str(exc) == "invalid top_logprobs"
    else:
        raise AssertionError("expected ValueError")


def test_missing_entropy_continues() -> None:
    assert decide(None, 0.5) == "continue"
    assert decide(0.5, 0.5) == "cut"
    assert decide(0.5001, 0.5) == "continue"


def test_threshold_methods() -> None:
    values = thresholds([0.8, 0.6], [0.2, 0.4])
    assert math.isclose(values["mean_correct"], 0.7)
    assert math.isclose(
        values["zero_observed_error"], 0.2 - 1e-9, rel_tol=0, abs_tol=1e-15
    )


def test_complete_parses_mlx_content_logprobs(monkeypatch) -> None:
    payload = {
        "choices": [
            {
                "text": "answer",
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"logprob": math.log(0.75)},
                                {"logprob": math.log(0.25)},
                            ]
                        },
                        {
                            "top_logprobs": [
                                {"logprob": math.log(0.5)},
                                {"logprob": math.log(0.5)},
                            ]
                        },
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 2},
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        "mlx_gptq.reasoning_entropy.urllib.request.urlopen",
        lambda _request, timeout: Response(),
    )
    result = complete(
        "http://127.0.0.1:8083",
        "flint",
        "prompt",
        2,
        timeout=1,
        top_logprobs=20,
    )
    assert result.content == "answer"
    assert result.tokens == 2
    assert result.finish_reason == "stop"
    assert result.entropy is not None


def test_final_answer_after_think_close() -> None:
    assert final_answer("reasoning</think>answer") == "answer"
