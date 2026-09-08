"""FAISS indexes partitioned by embedding model identity."""

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
    """Store embeddings and their document metadata for semantic search."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        index_path: str | Path | None = None,
        require_identity: bool = False,
    ) -> None:
        self._dim = dim
        self._index_path = str(index_path or (rune_data() / "vectors"))
        self._base_index_path = self._index_path
        self._require_identity = require_identity
        self._fingerprint: str | None = None
        self._index: Any = None  # faiss.IndexHNSWFlat
        self._metadata: list[VectorMetadata] = []
        self._initialized = False
        self._dirty = False
        self._deleted_count = 0

    def _bind_embedding(self, embedding: list[float]) -> None:
        fingerprint = getattr(embedding, "fingerprint", None)
        if not fingerprint:
            if self._require_identity:
                raise ValueError("Vector provenance is missing")
            return
        self.select_model(fingerprint, len(embedding))

    def select_model(self, fingerprint: str, dimensions: int) -> None:
        """Select a derived cache without requiring a new model inference."""
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("Invalid vector fingerprint")
        if not isinstance(dimensions, int) or not 1 <= dimensions <= 8192:
            raise ValueError("Invalid vector dimensions")
        if self._fingerprint == fingerprint:
            if self._dim != dimensions:
                raise ValueError("Same model identity returned different dimensions")
            return
        if self._dirty:
            self._save()
        self._fingerprint = fingerprint
        self._index_path = str(Path(self._base_index_path) / fingerprint)
        self._dim = dimensions
        self._index = None
        self._metadata = []
        self._initialized = False
        self._dirty = False
        self._deleted_count = 0

    def indexed_ids(self, types: set[str] | None = None) -> set[str]:
        self._ensure_init()
        return {meta.id for meta in self._metadata
                if meta.type != "_deleted" and (types is None or meta.type in types)}

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        if self._require_identity and self._fingerprint is None:
            raise ValueError("Embedding model is not ready")

        try:
            import faiss  # type: ignore[import-untyped]  # noqa: F811
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
                if self._index.d != self._dim:
                    raise ValueError("Stored vector dimension differs from the active model")
                with open(meta_file) as f:
                    raw = json_decode(f.read())
                self._metadata = [VectorMetadata(**m) for m in raw]
                if len(self._metadata) != self._index.ntotal:
                    raise ValueError("Vector index and metadata counts differ")
                self._deleted_count = sum(m.type == "_deleted" for m in self._metadata)
                self._initialized = True
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
        self._deleted_count = 0
        self._initialized = True
        log.info("vector_index_created", dim=self._dim)

    def _compact(self) -> None:
        if self._deleted_count < 64 or self._deleted_count * 4 < len(self._metadata):
            return
        import faiss

        live = [i for i, meta in enumerate(self._metadata) if meta.type != "_deleted"]
        replacement = faiss.IndexHNSWFlat(self._dim, 32)
        replacement.hnsw.efConstruction = 40
        replacement.hnsw.efSearch = 64
        if live:
            replacement.add(self._index.reconstruct_batch(np.asarray(live, dtype=np.int64)))
        self._index = replacement
        self._metadata = [self._metadata[i] for i in live]
        self._deleted_count = 0
        self._dirty = True

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        self._compact()
        os.makedirs(self._index_path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(self._index_path, "index.faiss"))

        meta_dicts = [
            {"type": m.type, "id": m.id, "timestamp": m.timestamp,
             "summary": m.summary, "category": m.category}
            for m in self._metadata
        ]
        with open(os.path.join(self._index_path, "metadata.json"), "w") as f:
            f.write(json_encode(meta_dicts))
        self._dirty = False

    def add(self, embedding: list[float], metadata: VectorMetadata) -> None:
        """Add a single vector with metadata."""
        self._bind_embedding(embedding)
        self._ensure_init()
        if len(embedding) != self._dim:
            log.warning("embedding_dim_mismatch", got=len(embedding), expected=self._dim)
            return
        vec = np.array([embedding], dtype=np.float32)
        if not np.isfinite(vec).all():
            log.warning("embedding_contains_nan_inf")
            return
        self._index.add(vec)
        self._metadata.append(metadata)
        self._dirty = True

    def upsert(self, embedding: list[float], metadata: VectorMetadata) -> None:
        self._bind_embedding(embedding)
        self._ensure_init()
        if len(embedding) != self._dim or not np.isfinite(np.asarray(embedding, dtype=np.float32)).all():
            raise ValueError("Invalid replacement vector")
        self.delete_by_id(metadata.id)
        self.add(embedding, metadata)
        self._compact()

    def add_batch(
        self, embeddings: list[list[float]], metadata_list: list[VectorMetadata],
    ) -> None:
        """Add multiple vectors at once."""
        if not embeddings:
            return
        if len(embeddings) != len(metadata_list):
            raise ValueError("Vector and metadata counts differ")
        if len({getattr(v, "fingerprint", None) for v in embeddings}) != 1:
            raise ValueError("Embedding batch mixes models")
        self._bind_embedding(embeddings[0])
        self._ensure_init()
        valid_embeddings = []
        valid_metadata = []
        for emb, meta in zip(embeddings, metadata_list, strict=False):
            if len(emb) != self._dim:
                log.warning("embedding_dim_mismatch", got=len(emb), expected=self._dim)
                continue
            arr = np.array(emb, dtype=np.float32)
            if not np.isfinite(arr).all():
                log.warning("embedding_contains_nan_inf")
                continue
            valid_embeddings.append(emb)
            valid_metadata.append(meta)
        if not valid_embeddings:
            return
        vecs = np.array(valid_embeddings, dtype=np.float32)
        self._index.add(vecs)
        self._metadata.extend(valid_metadata)
        self._dirty = True

    def delete_by_id(self, doc_id: str) -> bool:
        """Mark a vector as deleted by its document ID. Returns True if found."""
        self._ensure_init()
        found = False
        for i, meta in enumerate(self._metadata):
            if meta.id == doc_id and meta.type != "_deleted":
                self._metadata[i] = VectorMetadata(type="_deleted", id=doc_id)
                self._deleted_count += 1
                found = True
        self._dirty |= found
        return found

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
        self._bind_embedding(query_embedding)
        self._ensure_init()
        if k <= 0 or self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        if query.shape[1] != self._dim or not np.isfinite(query).all():
            raise ValueError("Invalid query vector")
        import faiss

        fetch_k = min(k * 3 if type_filter else k, self._index.ntotal)
        while True:
            params = faiss.SearchParametersHNSW(efSearch=max(64, fetch_k))
            distances, indices = self._index.search(query, fetch_k, params=params)
            results = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                meta = self._metadata[idx]
                if meta.type == "_deleted" or (type_filter is not None and meta.type != type_filter):
                    continue
                results.append(SearchResult(id=meta.id, score=1.0 / (1.0 + float(dist)),
                                            metadata=meta, text=meta.summary))
            if len(results) >= k or fetch_k == self._index.ntotal:
                return results[:k]
            fetch_k = min(fetch_k * 2, self._index.ntotal)

    def save(self) -> None:
        """Persist the index to disk."""
        if self._initialized:
            self._save()

    @property
    def count(self) -> int:
        self._ensure_init()
        return int(self._index.ntotal)

    def clear(self) -> None:
        """Remove all vectors and reset the index."""
        self._initialized = False
        self._index = None
        self._metadata = []
        self._dirty = False
        self._deleted_count = 0
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
        _vector_store = VectorStore(require_identity=True)
    return _vector_store
