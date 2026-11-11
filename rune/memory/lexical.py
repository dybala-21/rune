"""Keyword scoring for memory: BM25 over the character bigrams of each word.

Whole-word matching misses most Korean matches, because particles and endings attach to the noun
(커피를, 커피는), and it misses English inflections too (database, databases). The bigrams of each
word match across both, the same way in every language, with no per-language rules. On
Ko-StrategyQA (592 queries, 9,251 passages) recall@10 went from 0.20 with whitespace word overlap
to 0.70.

This is the keyword side of memory search, and what memory search falls back to when embeddings
can't answer.
"""

from __future__ import annotations

import functools
import heapq
import math
from collections import Counter
from collections.abc import Hashable

K1 = 1.2
B = 0.75


def terms(text: str) -> list[str]:
    """The character bigrams of each word; a one-character word is its own term."""
    out: list[str] = []
    for word in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split():
        out += [word] if len(word) < 2 else [word[i:i + 2] for i in range(len(word) - 1)]
    return out


class LexicalIndex[K: Hashable]:
    """BM25 over a changing set of documents, keyed by id. Adding an id again replaces it."""

    def __init__(self) -> None:
        self._docs: dict[K, Counter[str]] = {}
        self._lengths: dict[K, int] = {}
        self._postings: dict[str, dict[K, int]] = {}
        self._total = 0

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, key: object) -> bool:
        return key in self._docs

    def add(self, key: K, text: str) -> None:
        self.remove(key)
        counts = Counter(terms(text))
        self._docs[key] = counts
        self._lengths[key] = length = sum(counts.values())
        self._total += length
        for term, n in counts.items():
            self._postings.setdefault(term, {})[key] = n

    def remove(self, key: K) -> None:
        counts = self._docs.pop(key, None)
        if counts is None:
            return
        self._total -= self._lengths.pop(key)
        for term in counts:
            posting = self._postings[term]
            del posting[key]
            if not posting:
                del self._postings[term]

    def scores(self, query: str) -> dict[K, float]:
        """BM25 of every document that shares a term with the query; the others score 0."""
        n = len(self._docs)
        if not n:
            return {}
        average = self._total / n or 1.0
        out: dict[K, float] = {}
        for term in set(terms(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for key, tf in posting.items():
                norm = K1 * (1 - B + B * self._lengths[key] / average)
                out[key] = out.get(key, 0.0) + idf * tf * (K1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int) -> list[tuple[K, float]]:
        return heapq.nlargest(k, self.scores(query).items(), key=lambda item: item[1])


def rank(query: str, texts: list[str]) -> list[float]:
    """BM25 of each text against the query, with the texts themselves as the corpus."""
    scores = _index_of(tuple(texts)).scores(query)
    return [scores.get(i, 0.0) for i in range(len(texts))]


@functools.lru_cache(maxsize=8)
def _index_of(texts: tuple[str, ...]) -> LexicalIndex[int]:
    # Callers rank the same fact list turn after turn: build its index once (3,000 facts: ~30 ms).
    index: LexicalIndex[int] = LexicalIndex()
    for i, text in enumerate(texts):
        index.add(i, text)
    return index
