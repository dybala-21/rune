"""Tests for rune.attachments.vision_cache — LRU cache and summary extraction."""

from __future__ import annotations

import re

from rune.attachments.vision_cache import (
    VisionCache,
    extract_image_summary,
)

# ---------------------------------------------------------------------------
# VisionCache — core operations
# ---------------------------------------------------------------------------


class TestVisionCache:
    def test_returns_none_on_miss(self):
        cache = VisionCache()
        assert cache.get("nonexistent") is None

    def test_hash_returns_16_char_hex(self):
        h = VisionCache.hash(b"test image data")
        assert len(h) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", h)

    def test_same_hash_for_same_buffer(self):
        buf = b"identical content"
        assert VisionCache.hash(buf) == VisionCache.hash(buf)

    def test_different_hash_for_different_buffers(self):
        h1 = VisionCache.hash(b"image A")
        h2 = VisionCache.hash(b"image B")
        assert h1 != h2

    def test_cache_and_retrieve(self):
        cache = VisionCache()
        cache.set("abc123", "Screenshot shows an error message.", 1843)
        result = cache.get("abc123")
        assert result is not None
        assert "Screenshot shows an error message." in result
        assert "Previously analysed" in result

    def test_tracks_hit_count(self):
        cache = VisionCache()
        cache.set("hash1", "summary text", 1000)
        cache.get("hash1")
        cache.get("hash1")
        result = cache.get("hash1")
        assert result is not None
        assert "#3" in result  # 3rd hit

    def test_stats(self):
        cache = VisionCache()
        cache.set("h1", "summary", 2000)
        cache.get("h1")
        cache.get("h1")
        stats = cache.stats()
        assert stats.entries == 1
        assert stats.total_hits == 2
        assert stats.total_tokens_saved > 0

    def test_invalidate(self):
        cache = VisionCache()
        cache.set("h1", "summary", 2000)
        assert cache.has("h1") is True
        cache.invalidate("h1")
        assert cache.has("h1") is False
        assert cache.get("h1") is None

    def test_lru_eviction(self):
        cache = VisionCache(max_entries=3)
        cache.set("a", "summary a", 100)
        cache.set("b", "summary b", 100)
        cache.set("c", "summary c", 100)
        cache.set("d", "summary d", 100)  # should evict 'a'
        assert cache.has("a") is False
        assert cache.has("b") is True
        assert cache.has("c") is True
        assert cache.has("d") is True

    def test_lru_keeps_recently_accessed(self):
        cache = VisionCache(max_entries=3)
        cache.set("a", "summary a", 100)
        cache.set("b", "summary b", 100)
        cache.set("c", "summary c", 100)
        cache.get("a")  # touch 'a'
        cache.set("d", "summary d", 100)  # should evict 'b'
        assert cache.has("a") is True
        assert cache.has("b") is False
        assert cache.has("c") is True
        assert cache.has("d") is True

    def test_does_not_cache_empty_summary(self):
        cache = VisionCache()
        cache.set("h1", "", 100)
        assert cache.has("h1") is False

    def test_truncates_long_summaries(self):
        cache = VisionCache()
        long_summary = "x" * 1000
        cache.set("h1", long_summary, 100)
        result = cache.get("h1")
        assert result is not None
        assert len(result) < 600

    def test_clear(self):
        cache = VisionCache()
        cache.set("a", "summary", 100)
        cache.set("b", "summary", 100)
        cache.clear()
        assert cache.stats().entries == 0


# ---------------------------------------------------------------------------
# extract_image_summary
# ---------------------------------------------------------------------------


class TestExtractImageSummary:
    def test_extracts_first_two_sentences(self):
        text = "This image shows an error log. A TypeError occurred. Third sentence."
        result = extract_image_summary(text)
        assert result is not None
        assert "error log." in result
        assert "TypeError occurred." in result
        assert "Third sentence" not in result

    def test_text_without_sentence_markers(self):
        text = "Screenshot showing a login form with an error"
        result = extract_image_summary(text)
        assert result == text

    def test_empty_string_returns_none(self):
        assert extract_image_summary("") is None

    def test_truncates_long_text_without_sentences(self):
        text = "a" * 200
        result = extract_image_summary(text)
        assert result is not None
        assert len(result) <= 150

    def test_handles_korean_sentence_endings(self):
        text = "\uc774\ubbf8\uc9c0\uc5d0 \ubc84\uadf8\uac00 \uc788\uc2b5\ub2c8\ub2e4. \uc218\uc815\uc774 \ud544\uc694\ud569\ub2c8\ub2e4."
        result = extract_image_summary(text)
        assert result is not None
        assert "\ubc84\uadf8\uac00 \uc788\uc2b5\ub2c8\ub2e4." in result
