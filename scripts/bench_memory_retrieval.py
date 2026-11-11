#!/usr/bin/env python3
"""Memory retrieval on Ko-StrategyQA dev (592 Korean queries, 9,251 passages): recall@10 and
MRR@10 for the keyword index memory search uses, and optionally for the embedding provider.

The keyword side is what hosted machines answer with when no embeddings run, and what every
machine falls back to. Measured 2026-11-10: whitespace word overlap 0.20, BM25 over character
bigrams 0.70 (the KeywordIndex now). The dataset (Apache-2.0, ~14 MB) is downloaded once into
the cache directory.

Usage:
    uv run python scripts/bench_memory_retrieval.py [--embeddings] [--cache DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/taeminlee/Ko-StrategyQA/resolve/main/"
FILES = {"corpus.jsonl": "corpus.jsonl", "queries.jsonl": "queries.jsonl", "dev.jsonl": "qrels/dev.jsonl"}


def load(cache: Path) -> tuple[list[str], list[str], list[str], list[set[str]]]:
    cache.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        if not (cache / name).exists():
            urllib.request.urlretrieve(BASE + remote, cache / f"{name}.part")  # noqa: S310 (fixed https URL)
            (cache / f"{name}.part").rename(cache / name)
    qrels: dict[str, set[str]] = {}
    for line in (cache / "dev.jsonl").read_text().splitlines():
        row = json.loads(line)
        qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    queries = {(r := json.loads(line))["_id"]: r["text"] for line in (cache / "queries.jsonl").read_text().splitlines()}
    qids = [q for q in qrels if q in queries]
    corpus = [json.loads(line) for line in (cache / "corpus.jsonl").read_text().splitlines()]
    ids = [c["_id"] for c in corpus]
    docs = [(c.get("title", "") + " " + c["text"]).strip()[:1500] for c in corpus]
    return [queries[q] for q in qids], ids, docs, [qrels[q] for q in qids]


def score(ranked: list[list[str]], relevant: list[set[str]]) -> tuple[float, float]:
    recall = sum(len(rel & set(top[:10])) / len(rel) for top, rel in zip(ranked, relevant, strict=True))
    mrr = sum(next((1 / (n + 1) for n, i in enumerate(top[:10]) if i in rel), 0.0)
              for top, rel in zip(ranked, relevant, strict=True))
    return round(recall / len(ranked), 3), round(mrr / len(ranked), 3)


def keyword(queries: list[str], ids: list[str], docs: list[str]) -> tuple[list[list[str]], float]:
    from rune.memory.types import VectorMetadata
    from rune.memory.vector import KeywordIndex

    index = KeywordIndex(max_items=len(docs))
    for i, d in zip(ids, docs, strict=True):
        index.add(d, VectorMetadata(type="passage", id=i, summary=""))
    began = time.perf_counter()
    ranked = [[r.id for r in index.search(q, k=10)] for q in queries]
    return ranked, (time.perf_counter() - began) * 1000 / len(queries)


async def embedded(queries: list[str], ids: list[str], docs: list[str]) -> list[list[str]]:
    import numpy as np

    from rune.llm.local_embedding import get_embedding_provider

    provider = get_embedding_provider()

    async def embed(texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 64):
            out += await provider.embed(texts[start:start + 64])
        a = np.asarray(out, dtype=np.float32)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    q, d = await embed(queries), await embed(docs)
    return [[ids[j] for j in np.argsort(-row)[:10]] for row in q @ d.T]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embeddings", action="store_true", help="also rank with the embedding provider")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "rune-bench" / "ko-strategyqa")
    args = parser.parse_args()
    queries, ids, docs, relevant = load(args.cache)
    ranked, per_query_ms = keyword(queries, ids, docs)
    r, m = score(ranked, relevant)
    print(f"keyword (BM25 bigram)  recall@10 {r:.3f}  MRR@10 {m:.3f}  {per_query_ms:.1f} ms/query")
    if args.embeddings:
        r, m = score(asyncio.run(embedded(queries, ids, docs)), relevant)
        print(f"embedding provider     recall@10 {r:.3f}  MRR@10 {m:.3f}")


if __name__ == "__main__":
    main()
