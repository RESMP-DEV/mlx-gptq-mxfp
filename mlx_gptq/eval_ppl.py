"""Quick perplexity eval for MLX models (Mac): compare RTN vs calibrated.

    python -m mlx_gptq.eval_ppl --model ./out-gptq --model ./out-rtn \
        --text ~/wikitext_test.txt --seqlen 1024 --windows 16
"""

from __future__ import annotations

import argparse
import math


def ppl(model_path: str, text: str, seqlen: int, windows: int) -> float:
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(model_path)
    ids = tokenizer.encode(text)
    n = min(windows, max(1, (len(ids) - 1) // seqlen))
    total_nll, total_tok = 0.0, 0
    for i in range(n):
        chunk = ids[i * seqlen : (i + 1) * seqlen + 1]
        inp = mx.array(chunk[:-1])[None]
        tgt = mx.array(chunk[1:])[None]
        logits = model(inp).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(logprobs, tgt[..., None], axis=-1).sum()
        mx.eval(nll)
        total_nll += float(nll)
        total_tok += tgt.size
    return math.exp(total_nll / total_tok)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args(argv)

    with open(args.text) as f:
        text = f.read()
    for mp in args.model:
        print(f"{mp}: ppl = {ppl(mp, text, args.seqlen, args.windows):.4f}")


if __name__ == "__main__":
    main()
