"""FAISS-based vector store for RUNE.

Ported from src/memory/vector.ts - replaces Vectra O(n) linear scan
with FAISS HNSW O(log n) search. Key performance improvement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from rune.memory.types import SearchResult, VectorMetadata  # re-export for compat
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

# Default embedding dimension (nomic-embed-text v1.5)
DEFAULT_DIM = 768

__all__ = ["VectorMetadata", "SearchResult", "VectorStore", "KeywordIndex", "get_vector_store"]


class VectorStore:
    """FAISS HNSW-based vector store for semantic search.

    Key improvement over TS: O(log n) vs O(n) search time.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        index_path: str | Path | None = None,
    ) -> None:
        self._dim = dim
        self._index_path = str(index_path or (rune_data() / "vectors"))
        self._index: Any = None  # faiss.IndexHNSWFlat
        self._metadata: list[VectorMetadata] = []
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        try:
            import faiss  # noqa: F811
        except (ImportError, ModuleNotFoundError):
            raise RuntimeError(
                "faiss-cpu is not installed. Vector search requires it.\n"
                "Install with: pip install rune-ai[vector]"
            ) from None

        index_file = os.path.join(self._index_path, "index.faiss")
        meta_file = os.path.join(self._index_path, "metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self._index = faiss.read_index(index_file)
                with open(meta_file) as f:
                    raw = json_decode(f.read())
                self._metadata = [VectorMetadata(**m) for m in raw]
                log.info("vector_index_loaded", count=self._index.ntotal)
                return
            except Exception as exc:
                log.warning("vector_index_corrupt", error=str(exc))
                # Fall through to create new index

        # Create new HNSW index
        # M=32 (connections per layer), efConstruction=40
        self._index = faiss.IndexHNSWFlat(self._dim, 32)
        self._index.hnsw.efConstruction = 40
        self._index.hnsw.efSearch = 64
        self._metadata = []
        log.info("vector_index_created", dim=self._dim)

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        os.makedirs(self._index_path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(self._index_path, "index.faiss"))

        meta_dicts = [
            {"type": m.type, "id": m.id, "timestamp": m.timestamp,
             "summary": m.summary, "category": m.category}
            for m in self._metadata
        ]
        with open(os.path.join(self._index_path, "metadata.json"), "w") as f:
            f.write(json_encode(meta_dicts))

    def add(self, embedding: list[float], metadata: VectorMetadata) -> None:
        """Add a single vector with metadata."""
        self._ensure_init()
        vec = np.array([embedding], dtype=np.float32)
        self._index.add(vec)
        self._metadata.append(metadata)

    def add_batch(
        self, embeddings: list[list[float]], metadata_list: list[VectorMetadata],
    ) -> None:
        """Add multiple vectors at once."""
        self._ensure_init()
        if not embeddings:
            return
        vecs = np.array(embeddings, dtype=np.float32)
        self._index.add(vecs)
        self._metadata.extend(metadata_list)

    def delete_by_id(self, doc_id: str) -> bool:
        """Mark a vector as deleted by its document ID. Returns True if found."""
        for i, meta in enumerate(self._metadata):
            if meta.id == doc_id:
                # Mark as tombstone - set metadata type to "_deleted"
                self._metadata[i] = VectorMetadata(type="_deleted", id=doc_id)
                return True
        return False

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        type_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for the k nearest vectors.

        Returns results sorted by similarity (highest first).
        If *type_filter* is provided, only results whose metadata.type
        matches the given value are returned.
        """
        self._ensure_init()
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        # Fetch extra candidates when filtering so we still return up to k
        fetch_k = min(k * 3 if type_filter else k, self._index.ntotal)
        distances, indices = self._index.search(query, fetch_k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._metadata):
                continue
            # Convert L2 distance to similarity score (0-1)
            score = 1.0 / (1.0 + float(dist))
            meta = self._metadata[idx]
            results.append(SearchResult(
                id=meta.id,
                score=score,
                metadata=meta,
                text=meta.summary if meta else "",
            ))

        # Filter out tombstoned entries
        results = [r for r in results if r.metadata.type != "_deleted"]

        if type_filter is not None:
            results = [r for r in results if r.metadata and r.metadata.type == type_filter]

        return results[:k]

    def save(self) -> None:
        """Persist the index to disk."""
        if self._initialized:
            self._save()

    @property
    def count(self) -> int:
        self._ensure_init()
        return self._index.ntotal

    def clear(self) -> None:
        """Remove all vectors and reset the index."""
        self._initialized = False
        self._index = None
        self._metadata = []
        # Delete files
        for name in ("index.faiss", "metadata.json"):
            p = os.path.join(self._index_path, name)
            if os.path.exists(p):
                os.unlink(p)


# Keyword fallback search (for when vector search is unavailable)

class KeywordIndex:
    """Simple in-memory keyword search as fallback."""

    def __init__(self, max_items: int = 5000) -> None:
        self._items: list[tuple[str, VectorMetadata]] = []
        self._max = max_items

    def add(self, text: str, metadata: VectorMetadata) -> None:
        if len(self._items) >= self._max:
            self._items.pop(0)
        self._items.append((text.lower(), metadata))

    def search(
        self, query: str, k: int = 5, type_filter: str | None = None,
    ) -> list[SearchResult]:
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored: list[tuple[float, VectorMetadata]] = []
        for text, meta in self._items:
            if type_filter is not None and meta.type != type_filter:
                continue
            score = sum(1.0 for term in query_terms if term in text) / max(len(query_terms), 1)
            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(id=meta.id, score=score, metadata=meta, text=meta.summary if meta else "")
            for score, meta in scored[:k]
        ]


# Module singleton

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
