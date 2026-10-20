#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow==21.0.0"]
# ///
"""Build a deterministic, transformed Ultra-FineWeb long-context holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from heapq import heappush, heapreplace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

WORD_RE = re.compile(r"\w+", re.UNICODE)
SOURCE_REPOSITORY = "openbmb/Ultra-FineWeb-L1"
SOURCE_REVISION = "10b9ba18466215c0ba495299dfffd798af1027f2"
SOURCE_CONFIG = "CC-MAIN-2025-51"
SOURCE_SHARDS = {
    "CC-MAIN-2025-51-part-0001-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0001-of-1000.parquet",
        "dfbf0c78137b30ff4de5bd036d0306139b224da76729cffd1e1b15102500e976",
    ),
    "CC-MAIN-2025-51-part-0002-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0002-of-1000.parquet",
        "0ca959d2587f7bb534a2ca04a927529a4bcced4406f93719e14acac9d1208e8b",
    ),
    "CC-MAIN-2025-51-part-0003-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0003-of-1000.parquet",
        "177bfae4bb2ead0204e27dc78791c99b9be0e2f2e43652859234b6ea8151cceb",
    ),
    "CC-MAIN-2025-51-part-0004-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0004-of-1000.parquet",
        "89cd7ded294f51b7ab76ebf1200e16bdc9bcff975d0ac933df6c3229cca68615",
    ),
    "CC-MAIN-2025-51-part-0005-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0005-of-1000.parquet",
        "f969ee653d2ab20309db63225a688e3eafd3ad6de20d2ef3c44cccead2f169cb",
    ),
    "CC-MAIN-2025-51-part-0006-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0006-of-1000.parquet",
        "16c6f98a3b17ab58b95bc488e280a281a4d31607496dddedf951c93595863ac9",
    ),
    "CC-MAIN-2025-51-part-0007-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0007-of-1000.parquet",
        "3bf7b53edf1e51b9c612493120bccfd59aa2ebdc70bfe0914d87b47a0184323c",
    ),
    "CC-MAIN-2025-51-part-0008-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0008-of-1000.parquet",
        "33734bd826d0093c8b30a7072270a4aa591ccd15466c7471c16bc395464c606a",
    ),
    "CC-MAIN-2025-51-part-0009-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0009-of-1000.parquet",
        "2284bfa23822db54b662dd3d33804f0c75de18b92abb0501923636661f4d3f63",
    ),
    "CC-MAIN-2025-51-part-0010-of-1000.parquet": (
        "data/CC-MAIN-2025-51/CC-MAIN-2025-51-part-0010-of-1000.parquet",
        "a4303bbba310affa56a3babc79e7eeda4bf1b81047a28a3d8526b64088a7df6b",
    ),
}
LEXICAL_TECH_TERMS = {
    "software_engineering": (
        "software",
        "developer",
        "source code",
        "programming",
        "software developer",
        "application development",
        "web development",
        "coding",
        "codebase",
        "compiler",
        "debugging",
        "unit test",
        "repository",
        "github",
        "algorithm",
        "computer science",
        "python",
        "javascript",
        "typescript",
        "rust language",
        "golang",
        "c++",
    ),
    "systems": (
        "computer hardware",
        "linux",
        "linux kernel",
        "operating system",
        "file system",
        "filesystem",
        "computer memory",
        "process management",
        "multithreading",
        "network protocol",
        "computer network",
        "server",
        "processor",
        "memory allocation",
        "cpu architecture",
        "command line",
        "terminal",
        "device driver",
        "gpu",
        "cuda",
    ),
    "infrastructure": (
        " api",
        "cloud",
        "infrastructure",
        "api endpoint",
        "api request",
        "software development kit",
        "web hosting",
        "data center",
        "data pipeline",
        "docker",
        "kubernetes",
        "containerization",
        "cloud computing",
        "cloud infrastructure",
        "software deployment",
        "deployment",
        "devops",
        "continuous integration",
        "terraform",
        "amazon web services",
        "microsoft azure",
        "database query",
        "database",
        "sql query",
        "postgresql",
        "redis",
        "rest api",
    ),
    "security": (
        "cybersecurity",
        "vulnerability",
        "authentication",
        "authorization",
        "encryption",
        "malware",
        "threat model",
        "penetration testing",
        "security patch",
        "zero-day",
        " cve-",
    ),
    "ai_ml": (
        "artificial intelligence",
        "generative ai",
        "data science",
        "machine learning",
        "deep learning",
        "neural network",
        "transformer model",
        "large language model",
        "language model",
        " llm",
        "model inference",
        "model training",
        "embedding model",
        "vector database",
        "retrieval augmented",
        "ai agent",
        "agentic",
    ),
}
CODE_MARKERS = (
    "```",
    "pip install ",
    "npm install ",
    "git clone ",
    "docker run ",
    "kubectl ",
    "\nimport ",
    "\ndef ",
    "\nclass ",
    "public static void ",
    "#!/usr/bin/",
    "curl -",
)
POSITIVE_QUERIES = {
    "software_engineering": (
        "software engineering documentation with source code, APIs, debugging, "
        "testing, compilers, programming languages, and repository maintenance"
    ),
    "systems": (
        "technical systems engineering material about Linux, operating systems, "
        "networking, filesystems, device drivers, CPU architecture, GPU, or CUDA"
    ),
    "infrastructure": (
        "technical infrastructure documentation about cloud platforms, containers, "
        "Kubernetes, deployment, databases, CI/CD, and production operations"
    ),
    "security": (
        "technical computer security material about vulnerabilities, authentication, "
        "encryption, malware analysis, threat models, and security patches"
    ),
    "ai_ml": (
        "machine learning engineering material about model training, inference, "
        "transformers, embeddings, retrieval, AI agents, and developer tooling"
    ),
}
NEGATIVE_QUERIES = (
    "celebrity gossip, entertainment news, sports reporting, and popular culture",
    "travel diary, food recipe, fashion, wedding, lifestyle, and personal blog",
    "political speech, court reporting, religion, biography, and general news",
    "sales marketing copy, local business advertisement, and product promotion",
)
PRIMARY_QUERY = (
    "agentic coding context containing software repositories, source code, issue "
    "investigation, debugging, tests, APIs, build systems, developer tools, systems "
    "engineering, cloud infrastructure, or machine learning implementation details"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def shingle_hashes(text: str, width: int) -> set[int]:
    words = normalized_words(text)
    if len(words) < width:
        return set()
    return {
        int.from_bytes(
            hashlib.blake2b(
                "\0".join(words[index : index + width]).encode(), digest_size=8
            ).digest(),
            "big",
        )
        for index in range(len(words) - width + 1)
    }


def technical_scores(content: str, url: str) -> dict[str, int]:
    searchable = f" {content.casefold()} {url.casefold()} "
    scores = {
        category: sum(term in searchable for term in terms)
        for category, terms in LEXICAL_TECH_TERMS.items()
    }
    scores["software_engineering"] += sum(marker in searchable for marker in CODE_MARKERS)
    return scores


def embedding_request(
    endpoint: str, model: str, texts: list[str], input_type: str
) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts, "input_type": input_type}).encode()
    request = Request(  # noqa: S310 - endpoint is an explicit operator-controlled URL
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=300) as response:
        body = json.load(response)
    return [row["embedding"] for row in sorted(body["data"], key=lambda row: row["index"])]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def add_semantic_scores(
    candidates: list[dict[str, Any]], endpoint: str, model: str, batch_size: int
) -> None:
    categories = list(POSITIVE_QUERIES)
    anchor_texts = [PRIMARY_QUERY]
    anchor_texts.extend(POSITIVE_QUERIES[category] for category in categories)
    anchor_texts.extend(NEGATIVE_QUERIES)
    anchors = embedding_request(endpoint, model, anchor_texts, "query")
    primary = anchors[0]
    positive = anchors[1 : len(categories) + 1]
    negative = anchors[len(categories) + 1 :]

    for start in range(0, len(candidates), batch_size):
        rows = candidates[start : start + batch_size]
        snippets = []
        for row in rows:
            content = str(row["content"])
            snippets.append(f"URL: {row['url']}\n\n{content[:3000]}\n\n{content[-1000:]}")
        embeddings = embedding_request(endpoint, model, snippets, "passage")
        for row, embedding in zip(rows, embeddings, strict=True):
            positive_scores = [dot(embedding, anchor) for anchor in positive]
            negative_score = max(dot(embedding, anchor) for anchor in negative)
            best = max(range(len(categories)), key=positive_scores.__getitem__)
            row["primary_technical_category"] = categories[best]
            row["primary_query_similarity"] = dot(embedding, primary)
            row["semantic_technical_score"] = positive_scores[best]
            row["positive_similarity"] = positive_scores[best]
            row["negative_similarity"] = negative_score
        print(
            f"embedded {min(start + batch_size, len(candidates))}/{len(candidates)}",
            file=sys.stderr,
        )


def content_is_usable(
    content: str, meta: dict[str, Any], calibration_shingles: set[int]
) -> tuple[bool, int]:
    if not 4_000 <= len(content) <= 80_000:
        return False, 0
    if float(meta.get("language_score", 0.0)) < 0.98:
        return False, 0
    pii_fields = (
        "pii_emails",
        "pii_ips",
        "pii_phones",
        "pii_id_cards",
        "pii_credit_cards",
    )
    if any(int(meta.get(field, 0)) for field in pii_fields):
        return False, 0
    visible = [char for char in content if not char.isspace()]
    if not visible or sum(char.isalpha() for char in visible) / len(visible) < 0.65:
        return False, 0
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.8:
        return False, 0
    overlap = len(shingle_hashes(content, 20) & calibration_shingles)
    return overlap == 0, overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True, nargs="+")
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--target-chars", type=int, default=4_500_000)
    parser.add_argument("--candidate-limit", type=int, default=8192)
    parser.add_argument("--seed", default="gemma4-long-context-holdout-v1")
    parser.add_argument("--verified-token-count", type=int)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--embedding-endpoint", default="http://127.0.0.1:1235/v1/embeddings")
    parser.add_argument(
        "--embedding-model", default="jina-code-embeddings-1.5b-block-gptq-mxfp4-32k"
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--minimum-semantic-score", type=float, default=0.30)
    args = parser.parse_args()
    tokenization_values = (
        args.verified_token_count,
        args.tokenizer,
        args.tokenizer_revision,
    )
    if any(value is not None for value in tokenization_values) and not all(
        value is not None for value in tokenization_values
    ):
        raise SystemExit(
            "verified token count, tokenizer, and tokenizer revision must be supplied together"
        )

    source_records = []
    for path in args.source_parquet:
        try:
            source_path, expected_sha256 = SOURCE_SHARDS[path.name]
        except KeyError as exc:
            raise SystemExit(f"unrecognized source shard: {path.name}") from exc
        source_sha256 = sha256_file(path)
        if source_sha256 != expected_sha256:
            raise SystemExit(f"source shard SHA-256 mismatch for {path}: {source_sha256}")
        source_records.append({"path": path, "source_path": source_path, "sha256": source_sha256})

    calibration = args.calibration_text.read_text(encoding="utf-8")
    calibration_shingles = shingle_hashes(calibration, 20)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    scanned = 0
    qualified = 0

    columns = ["uid", "content", "meta", "dataset_index"]
    for source_record in source_records:
        parquet = pq.ParquetFile(source_record["path"])
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            for row in batch.to_pylist():
                scanned += 1
                content = row["content"]
                meta = json.loads(row["meta"])
                usable, overlap = content_is_usable(content, meta, calibration_shingles)
                if not usable:
                    continue
                scores = technical_scores(content, str(meta.get("url", "")))
                technical_score = sum(scores.values())
                if technical_score < 1:
                    continue
                qualified += 1
                selection_key = int.from_bytes(
                    hashlib.sha256(f"{args.seed}\0{row['uid']}".encode()).digest()[:8],
                    "big",
                )
                candidate = {
                    "selection_key": f"{selection_key:016x}",
                    "uid": row["uid"],
                    "dataset_index": row["dataset_index"],
                    "content": content,
                    "content_sha256": sha256_bytes(content.encode()),
                    "content_chars": len(content),
                    "url": meta.get("url"),
                    "domain": (urlparse(str(meta.get("url", ""))).hostname or "").lower(),
                    "warc_date": meta.get("warc_date"),
                    "warc_record_id": meta.get("warc_record_id"),
                    "source_file": meta.get("source_file"),
                    "source_shard": source_record["source_path"],
                    "language_score": float(meta["language_score"]),
                    "shared_calibration_20_word_shingles": overlap,
                    "technical_score": technical_score,
                    "technical_categories": [
                        category for category, score in scores.items() if score > 0
                    ],
                }
                item = (-selection_key, str(row["uid"]), candidate)
                if len(candidates) < args.candidate_limit:
                    heappush(candidates, item)
                elif selection_key < -candidates[0][0]:
                    heapreplace(candidates, item)

    semantic_candidates = [item[2] for item in candidates]
    add_semantic_scores(
        semantic_candidates,
        args.embedding_endpoint,
        args.embedding_model,
        args.embedding_batch_size,
    )
    ordered = sorted(
        (
            row
            for row in semantic_candidates
            if float(row["semantic_technical_score"]) >= args.minimum_semantic_score
        ),
        key=lambda row: (-float(row["semantic_technical_score"]), row["selection_key"]),
    )
    selected: list[dict[str, Any]] = []
    domains: set[str] = set()
    total_chars = 0
    for candidate in ordered:
        domain = str(candidate["domain"])
        if not domain or domain in domains:
            continue
        candidate["ordinal"] = len(selected)
        candidate["cumulative_chars"] = total_chars + int(candidate["content_chars"])
        selected.append(candidate)
        domains.add(domain)
        total_chars = int(candidate["cumulative_chars"])
        if total_chars >= args.target_chars:
            break
    if total_chars < args.target_chars:
        raise SystemExit(
            f"only selected {total_chars:,} of {args.target_chars:,} target characters"
        )

    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(str(row["content"]) for row in selected) + "\n"
    args.output_text.write_text(text, encoding="utf-8")

    public_rows = []
    for row in selected:
        public_rows.append(
            {
                **row,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_config": SOURCE_CONFIG,
            }
        )
    pq.write_table(pa.Table.from_pylist(public_rows), args.output_manifest, compression="zstd")

    receipt = {
        "schema_version": 1,
        "purpose": "evaluation-only long-context holdout; never calibration input",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "config": SOURCE_CONFIG,
            "shards": [
                {
                    "path": record["source_path"],
                    "sha256": record["sha256"],
                }
                for record in source_records
            ],
        },
        "selection": {
            "seed": args.seed,
            "rows_scanned": scanned,
            "rows_passing_filters": qualified,
            "documents_selected": len(selected),
            "unique_domains": len(domains),
            "documents_by_primary_technical_category": {
                category: sum(row["primary_technical_category"] == category for row in selected)
                for category in POSITIVE_QUERIES
            },
            "target_chars": args.target_chars,
            "output_chars": len(text),
            "filters": {
                "content_chars": [4_000, 80_000],
                "minimum_language_score": 0.98,
                "maximum_documents_per_domain": 1,
                "minimum_alpha_fraction_of_non_whitespace": 0.65,
                "minimum_unique_nonempty_line_fraction": 0.8,
                "pii_counts": 0,
                "shared_20_word_shingles_with_calibration": 0,
                "minimum_lexical_technical_signal_score": 1,
                "minimum_semantic_score": args.minimum_semantic_score,
            },
        },
        "semantic_filter": {
            "endpoint": args.embedding_endpoint,
            "model": args.embedding_model,
            "model_source": "jinaai/jina-code-embeddings-1.5b",
            "model_source_revision": "39aeb4fb9b60f930934c78ae5d749a46287c248a",
            "model_weights_sha256": (
                "9cadd568e6757e15358d960e4688bb0792a7edcd9889ef3d37f2361f4db6b756"
            ),
            "positive_queries": POSITIVE_QUERIES,
            "negative_queries": NEGATIVE_QUERIES,
            "primary_query": PRIMARY_QUERY,
        },
        "calibration_exclusion": {
            "path": str(args.calibration_text),
            "sha256": sha256_file(args.calibration_text),
            "twenty_word_shingles": len(calibration_shingles),
        },
        "artifacts": {
            "local_text": {
                "path": str(args.output_text),
                "bytes": args.output_text.stat().st_size,
                "sha256": sha256_file(args.output_text),
                "published": True,
            },
            "selection_manifest": {
                "path": str(args.output_manifest),
                "bytes": args.output_manifest.stat().st_size,
                "sha256": sha256_file(args.output_manifest),
                "contains_source_text": True,
            },
        },
    }
    if args.verified_token_count is not None:
        receipt["tokenization"] = {
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "tokens_without_bos": args.verified_token_count,
        }
    args.output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
