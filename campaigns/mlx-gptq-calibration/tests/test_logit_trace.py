from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from mlx_gptq_calibration.logit_trace import build_commands, parse_kl_statistics


class LogitTraceTests(unittest.TestCase):
    def test_parses_llama_kl_summary(self) -> None:
        output = """
Mean PPL(Q)                   :  12.500000 ±   0.100000
Mean PPL(base)                :  12.000000 ±   0.090000
Mean PPL(Q)/PPL(base)         :   1.041667 ±   0.010000
Mean    KLD:   0.097880 ±   0.001000
Maximum KLD:   9.000000
99.9%   KLD:   2.708700
99.0%   KLD:   1.250000
Median  KLD:   0.010000
RMS Δp    :  2.500 ± 0.100 %
Same top p: 85.630 ± 0.200 %
"""
        stats = parse_kl_statistics(output)
        self.assertEqual(stats["mean_kl_divergence"], 0.09788)
        self.assertEqual(stats["p99_9_kl_divergence"], 2.7087)
        self.assertEqual(stats["top1_agreement_percent"], 85.63)
        self.assertEqual(stats["trace_reference_perplexity_clipped"], 12.0)

    def test_commands_share_context_and_reference_trace(self) -> None:
        args = argparse.Namespace(
            llama_perplexity=Path("llama-perplexity"),
            reference=Path("reference.gguf"),
            candidate=Path("candidate.gguf"),
            corpus=Path("corpus.txt"),
            logits_file=Path("reference.bin"),
            ctx_size=512,
            batch_size=512,
            chunks=4,
            threads=8,
            gpu_layers=999,
            ubatch_size=None,
            flash_attn=None,
            no_kv_offload=False,
        )
        reference, candidate = build_commands(args)
        self.assertIn("--save-all-logits", reference)
        self.assertIn("--kl-divergence", candidate)
        self.assertEqual(reference[-12:], candidate[-12:])

    def test_long_context_runtime_flags_are_shared(self) -> None:
        args = argparse.Namespace(
            llama_perplexity=Path("llama-perplexity"),
            reference=Path("reference.gguf"),
            candidate=Path("candidate.gguf"),
            corpus=Path("corpus.txt"),
            logits_file=Path("reference.bin"),
            ctx_size=262144,
            batch_size=4096,
            chunks=1,
            threads=8,
            gpu_layers=20,
            ubatch_size=512,
            flash_attn="on",
            no_kv_offload=True,
        )
        reference, candidate = build_commands(args)
        for flag in ("--ubatch-size", "--flash-attn", "--no-kv-offload"):
            self.assertIn(flag, reference)
            self.assertIn(flag, candidate)


if __name__ == "__main__":
    unittest.main()
