"""Render a curated, provenance-rich KernelBench-Hard calibration stratum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SELECTED_RUN_IDS = (
    # RTX PRO 6000: one long, high-information trace per kernel family.
    "20260614_145529_zai-claude_glm-5.2_01_fp8_gemm",
    "20260613_163858_kimi-claude_kimi-k2.7-code_02_kda_cutlass",
    "20260613_055815_zai-claude_glm-5.2_03_paged_attention",
    "20260615_132230_deepseek-claude_deepseek-v4-pro_05_topk_bitonic",
    "20260613_171946_kimi-claude_kimi-k2.7-code_06_sonic_moe_swiglu",
    "20260613_061443_zai-claude_glm-5.2_07_w4a16_gemm",
    # Cross-architecture contrasts on H100 and B200.
    "20260618_060247_kimi-claude_kimi-k2.7-code_01_fp8_gemm",
    "20260618_060202_deepseek-claude_deepseek-v4-pro_05_topk_bitonic",
    "20260618_065423_minimax-claude_MiniMax-M3_07_w4a16_gemm",
    "20260618_204323_deepseek-claude_deepseek-v4-pro_02_kda_cutlass",
    "20260618_222009_zai-claude_glm-5.2_03_paged_attention",
    "20260618_222339_zai-claude_glm-5.2_06_sonic_moe_swiglu",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stringify(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_block(block: dict) -> tuple[str, str | None]:
    kind = block.get("type", "unknown")
    if kind == "thinking":
        return f"<assistant_thinking>\n{stringify(block.get('thinking', ''))}\n</assistant_thinking>", None
    if kind == "text":
        return f"<assistant_text>\n{stringify(block.get('text', ''))}\n</assistant_text>", None
    if kind == "tool_use":
        name = block.get("name", "unknown")
        payload = stringify(block.get("input", {}))
        return f'<tool_call name="{name}">\n{payload}\n</tool_call>', name
    if kind == "tool_result":
        content = stringify(block.get("content", ""))
        return f"<tool_result>\n{content}\n</tool_result>", None
    return f'<message_block type="{kind}">\n{stringify(block)}\n</message_block>', None


def render_trace(path: Path, metadata: dict) -> tuple[str, dict]:
    raw = path.read_bytes()
    blocks = [
        "<kernelbench_trace\n"
        f"run_id={metadata['run_id']}\n"
        f"gpu={metadata['gpu']}\n"
        f"model={metadata['model']}\n"
        f"problem={metadata['problem']}\n"
        f"correct={metadata['correct']}\n"
        f"peak_fraction={metadata['peak_fraction']}\n"
        ">"
    ]
    counts = Counter()
    tool_counts = Counter()
    for line in raw.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                blocks.append(f"<{role}>\n{content}\n</{role}>")
                counts[f"role:{role}"] += 1
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            rendered, tool_name = render_block(block)
            if rendered.strip():
                blocks.append(rendered)
                counts[f"block:{block.get('type', 'unknown')}"] += 1
            if tool_name:
                tool_counts[tool_name] += 1
    blocks.append("</kernelbench_trace>")
    text = "\n\n".join(blocks)
    receipt = {
        **metadata,
        "source_path": str(path.resolve()),
        "source_bytes": len(raw),
        "source_sha256": digest(raw),
        "rendered_blocks": len(blocks),
        "message_counts": dict(sorted(counts.items())),
        "tool_calls": dict(sorted(tool_counts.items())),
    }
    return text, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.receipt.exists():
        parser.error("output paths must not already exist")

    manifest_bytes = args.manifest.read_bytes()
    rows = {
        row["run_id"]: row
        for row in csv.DictReader(manifest_bytes.decode().splitlines())
    }
    missing = [run_id for run_id in SELECTED_RUN_IDS if run_id not in rows]
    if missing:
        raise RuntimeError(f"selected run ids absent from manifest: {missing}")

    rendered = []
    records = []
    for run_id in SELECTED_RUN_IDS:
        path = args.source_dir / f"{run_id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        text, record = render_trace(path, rows[run_id])
        rendered.append(text)
        records.append(record)

    payload = "\n\n".join(rendered) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)

    model_tokens = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model_tokens = len(tokenizer(payload, add_special_tokens=False).input_ids)

    receipt = {
        "algorithm": "curated-open-reasoning-kernelbench-hard-render-v1",
        "purpose": "post-training quantization activation calibration only; never fine-tuning",
        "source": {
            "dataset": "Infatoshi/kernelbench-hard-traces",
            "revision": args.dataset_revision,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": digest(manifest_bytes),
        },
        "selection": {
            "traces": len(records),
            "problems": dict(sorted(Counter(r["problem"] for r in records).items())),
            "gpus": dict(sorted(Counter(r["gpu"] for r in records).items())),
            "models": dict(sorted(Counter(r["model"] for r in records).items())),
            "correct": dict(sorted(Counter(r["correct"] for r in records).items())),
            "records": records,
        },
        "output": {
            "path": str(args.output.resolve()),
            "bytes": len(payload.encode()),
            "sha256": digest(payload.encode()),
            "model_tokens": model_tokens,
        },
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt["selection"], "records": "omitted", "output": receipt["output"]}, indent=2))


if __name__ == "__main__":
    main()
