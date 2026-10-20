from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "modal/bench_model.py"
SPEC = importlib.util.spec_from_file_location("bench_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_model)


class BenchModelTests(unittest.TestCase):
    def test_window_plan_scores_each_causal_token_once(self) -> None:
        windows = bench_model.window_plan(
            5000,
            seq_len=2048,
            stride=512,
            max_chunks=64,
        )
        self.assertEqual(
            windows,
            [
                (0, 2048, 2047),
                (512, 2560, 512),
                (1024, 3072, 512),
                (1536, 3584, 512),
                (2048, 4096, 512),
                (2560, 4608, 512),
                (3072, 5000, 392),
            ],
        )
        self.assertEqual(sum(scored for _, _, scored in windows), 4999)

    def test_window_chunk_cap_is_inclusive_and_counts_windows_exactly(self) -> None:
        windows = bench_model.window_plan(
            5000,
            seq_len=2048,
            stride=512,
            max_chunks=1,
        )
        self.assertEqual(windows, [(0, 2048, 2047)])

    def test_causal_slices_predict_the_final_scored_positions(self) -> None:
        self.assertEqual(
            bench_model.causal_score_slices(2048, 2047), (slice(0, 2047), slice(1, 2048))
        )
        self.assertEqual(
            bench_model.causal_score_slices(2048, 512), (slice(1535, 2047), slice(1536, 2048))
        )

    def test_causal_slices_reject_zero_scored_window(self) -> None:
        with self.assertRaisesRegex(bench_model.BenchError, "scored token count"):
            bench_model.causal_score_slices(2048, 2048)

    def test_window_plan_rejects_a_gap(self) -> None:
        with self.assertRaisesRegex(bench_model.BenchError, "stride"):
            bench_model.window_plan(
                5000,
                seq_len=1024,
                stride=2048,
                max_chunks=4,
            )


if __name__ == "__main__":
    unittest.main()
