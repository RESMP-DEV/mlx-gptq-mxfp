"""Calibration data loading.

Sources (comma-separated in the CLI):
    /path/file.txt                     raw text (e.g. calibration_datav3.txt)
    /path/file.jsonl[:field]           one JSON object per line, default field "text"
    hf:dataset[:split[:field]]         HuggingFace dataset, plain text field
    hfchat:dataset[#config][:split[:maxrows]]
        chat/agent dataset rendered through the model's OWN chat template
        (tool schemas included), so calibration covers the real tool-call
        token distribution. Streams, so huge datasets are fine with maxrows.
        Handles messages(role/content[/tool_calls]), ShareGPT
        conversations(from/value), and trajectory(role/text) schemas.

All documents are tokenized, joined with EOS, and nsamples random windows of
seqlen tokens are drawn from the concatenated stream (standard GPTQ practice).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

log = logging.getLogger(__name__)

_ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant",
    "system": "system",
    "tool": "tool", "function-response": "tool", "function_response": "tool",
    "function": "tool", "observation": "tool",
}


def _parse_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _normalize_messages(row) -> list | None:
    """Best-effort conversion of common agent-dataset schemas to
    [{role, content, [tool_calls]}, ...] for apply_chat_template."""
    msgs = row.get("messages") or row.get("conversations") or row.get("trajectory")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except json.JSONDecodeError:
            return None
    if not isinstance(msgs, list) or not msgs:
        return None
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = m.get("role") or m.get("from") or ""
        role = _ROLE_MAP.get(role.lower())
        if role is None:
            return None
        content = m.get("content") if m.get("content") is not None else (
            m.get("value") if m.get("value") is not None else m.get("text"))
        msg = {"role": role, "content": content if content is not None else ""}
        tc = m.get("tool_calls")
        if not tc and isinstance(m.get("function_call"), dict):
            fc = m["function_call"]
            tc = [{"type": "function",
                   "function": {"name": fc.get("name", ""),
                                "arguments": fc.get("arguments", {})}}]
        if tc:
            fixed = []
            for c in tc:
                c = dict(c)
                fn = dict(c.get("function", {}))
                fn["arguments"] = _parse_args(fn.get("arguments"))
                c["function"] = fn
                fixed.append(c)
            msg["tool_calls"] = fixed
        out.append(msg)
    return out


def _row_tools(row):
    tools = row.get("tools") or row.get("available_tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    return tools if isinstance(tools, list) and tools else None


def _read_chat_source(spec: str, tokenizer) -> list[str]:
    """hfchat:name[#config][:split[:maxrows]] -> rendered chat documents."""
    from datasets import load_dataset
    from itertools import islice

    parts = spec.split(":")
    name, config = parts[1], None
    if "#" in name:
        name, config = name.split("#", 1)
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    maxrows = int(parts[3]) if len(parts) > 3 and parts[3] else 2000

    ds = load_dataset(name, config, split=split, streaming=True)
    docs, skipped = [], 0
    for row in islice(ds, maxrows):
        msgs = _normalize_messages(row)
        if msgs is None:
            skipped += 1
            continue
        try:
            text = tokenizer.apply_chat_template(
                msgs, tools=_row_tools(row), tokenize=False
            )
        except Exception:
            skipped += 1
            continue
        docs.append(text)
    if skipped:
        log.warning("%s: skipped %d/%d rows (schema/template mismatch)",
                    spec, skipped, skipped + len(docs))
    if not docs:
        raise ValueError(f"{spec}: no rows could be rendered")
    return docs


def _read_source(spec: str, tokenizer=None) -> list[str]:
    if spec.startswith("hfchat:"):
        return _read_chat_source(spec, tokenizer)
    if spec.startswith("hf:"):
        parts = spec.split(":")
        name, split, field = parts[1], "train", "text"
        if len(parts) > 2 and parts[2]:
            split = parts[2]
        if len(parts) > 3 and parts[3]:
            field = parts[3]
        from datasets import load_dataset

        ds = load_dataset(name, split=split)
        return [r[field] for r in ds if r.get(field)]
    if ".jsonl" in spec:
        path, _, field = spec.partition(".jsonl")
        path += ".jsonl"
        field = field.lstrip(":") or "text"
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)[field])
        return docs
    with open(spec) as f:
        return [f.read()]


def load_calibration(
    tokenizer, sources: str, nsamples: int, seqlen: int, seed: int = 17
) -> torch.Tensor:
    docs = []
    for spec in sources.split(","):
        spec = spec.strip()
        if spec:
            got = _read_source(spec, tokenizer)
            log.info("dataset %s: %d documents", spec, len(got))
            docs.extend(got)
    if not docs:
        raise ValueError("no calibration documents loaded")

    eos = tokenizer.eos_token_id
    stream: list[int] = []
    for d in docs:
        stream.extend(tokenizer(d, add_special_tokens=False).input_ids)
        if eos is not None:
            stream.append(eos)
    ids = np.asarray(stream, dtype=np.int64)
    log.info("calibration stream: %d tokens", len(ids))

    if len(ids) < seqlen + 1:
        raise ValueError(f"corpus has {len(ids)} tokens < seqlen {seqlen}")

    rng = np.random.default_rng(seed)
    max_start = len(ids) - seqlen
    n_disjoint = max_start // seqlen
    if n_disjoint >= nsamples:
        starts = rng.permutation(n_disjoint)[:nsamples] * seqlen
    else:
        log.warning(
            "corpus only supports %d disjoint windows of %d tokens; sampling %d "
            "overlapping windows (consider more calibration data)",
            n_disjoint, seqlen, nsamples,
        )
        starts = rng.integers(0, max_start + 1, size=nsamples)

    return torch.from_numpy(np.stack([ids[s : s + seqlen] for s in sorted(starts)]))
