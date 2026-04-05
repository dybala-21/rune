"""Vision cache - LRU cache for vision API results with token savings tracking.

Ported from src/attachments/vision-cache.ts.

Workflow:
  1. On image attachment, compute sha256(buffer)[:16] as hash
  2. Cache miss: send full ImagePart to LLM; extract summary from response; store
  3. Cache hit: replace ImagePart with text summary (~50 tokens)
  4. "Look at the original again" -> invalidate() -> next turn re-sends full image

Savings example: 1920x1080 x 5 turns = 9,215 -> 2,043 tokens (78% reduction)
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import time
from dataclasses import dataclass

# Types


@dataclass(slots=True)
class VisionCacheEntry:
    summary: str
    original_tokens: int
    summary_tokens: int
    cached_at: float  # epoch seconds
    hit_count: int = 0


@dataclass(slots=True)
class VisionCacheStats:
    entries: int
    total_hits: int
    total_tokens_saved: int


# Constants

_DEFAULT_MAX_ENTRIES = 20
_SUMMARY_MAX_LENGTH = 500


# Token estimation (simple heuristic, avoids external dependency)

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


# VisionCache

class VisionCache:
    """LRU cache for vision API results with token savings tracking."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._entries: dict[str, VisionCacheEntry] = {}
        self._access_order: list[str] = []
        self._max_entries = max_entries
        self._total_hits = 0
        self._total_tokens_saved = 0

    @staticmethod
    def hash(data: bytes) -> str:
        """Compute sha256(data)[:16] for 64-bit collision resistance."""
        return hashlib.sha256(data).hexdigest()[:16]

    def get(self, hash_key: str) -> str | None:
        """Look up a cached summary by hash.

        On hit, increments counters and returns a formatted summary string.
        On miss, returns ``None``.
        """
        entry = self._entries.get(hash_key)
        if entry is None:
            return None

        entry.hit_count += 1
        self._total_hits += 1
        self._total_tokens_saved += entry.original_tokens - entry.summary_tokens
        self._touch_lru(hash_key)

        return (
            f"[Previously analysed image (ref #{entry.hit_count}): "
            f"{entry.summary}]"
        )

    def set(
        self,
        hash_key: str,
        summary: str,
        original_tokens: int,
    ) -> None:
        """Store a vision summary in the cache.

        Parameters
        ----------
        hash_key:
            The content hash from ``VisionCache.hash()``.
        summary:
            LLM-generated image description summary.
        original_tokens:
            Token cost of the original image (``estimated_tokens``).
        """
        if not summary:
            return

        trimmed = (
            summary[:_SUMMARY_MAX_LENGTH] + "..."
            if len(summary) > _SUMMARY_MAX_LENGTH
            else summary
        )
        summary_tokens = _estimate_tokens(trimmed)

        self._entries[hash_key] = VisionCacheEntry(
            summary=trimmed,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            cached_at=time.time(),
            hit_count=0,
        )
        self._touch_lru(hash_key)
        self._evict_if_needed()

    def invalidate(self, hash_key: str) -> None:
        """Force-remove a cache entry (e.g. user wants to re-examine image)."""
        self._entries.pop(hash_key, None)
        self._access_order = [h for h in self._access_order if h != hash_key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()
        self._access_order.clear()

    def has(self, hash_key: str) -> bool:
        """Check if a hash exists in the cache (without counting as a hit)."""
        return hash_key in self._entries

    def stats(self) -> VisionCacheStats:
        """Return cache statistics."""
        return VisionCacheStats(
            entries=len(self._entries),
            total_hits=self._total_hits,
            total_tokens_saved=self._total_tokens_saved,
        )

    def _touch_lru(self, hash_key: str) -> None:
        with contextlib.suppress(ValueError):
            self._access_order.remove(hash_key)
        self._access_order.append(hash_key)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries and self._access_order:
            oldest = self._access_order.pop(0)
            self._entries.pop(oldest, None)


# Summary extraction helper

def extract_image_summary(llm_response: str) -> str | None:
    """Extract an image-related summary from an LLM response.

    Takes the first 1-2 sentences (up to 300 chars).
    Returns ``None`` if the response is empty.
    """
    if not llm_response:
        return None

    # Try to extract first 2 sentences
    sentences = re.findall(r"[^.!?]+[.!?]", llm_response)
    if not sentences:
        # No sentence boundaries found: take first 150 chars
        text = llm_response[:150].strip() if len(llm_response) > 150 else llm_response.strip()
        return text or None

    summary = ""
    for sentence in sentences[:2]:
        if len(summary) + len(sentence) > 300:
            break
        summary += sentence

    return summary.strip() or None
