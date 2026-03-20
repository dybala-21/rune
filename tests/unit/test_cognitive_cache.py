"""Tests for the cognitive cache module."""

from __future__ import annotations

from rune.agent.cognitive_cache import (
    KnowledgeInventory,
    SessionToolCache,
    format_knowledge_inventory,
)


class TestSessionToolCache:
    def test_generate_key_file_read(self):
        cache = SessionToolCache()
        key = cache.generate_key("file_read", {"file_path": "/tmp/foo.py"})
        assert key is not None
        assert key.startswith("file_read:")
        assert "foo.py" in key or "tmp" in key

    def test_generate_key_uncacheable(self):
        cache = SessionToolCache()
        assert cache.generate_key("bash_execute", {"command": "ls"}) is None
        assert cache.generate_key("think", {"text": "hmm"}) is None
        # Browser and unknown capabilities are not cached
        assert cache.generate_key("browser_navigate", {"url": "http://x"}) is None

    def test_set_and_get(self):
        cache = SessionToolCache()
        key = cache.generate_key("file_read", {"file_path": "/tmp/a.py"})
        assert key is not None
        cache.set(key, "file_read", {"file_path": "/tmp/a.py"}, "content of file", step_number=1)
        hit = cache.get(key, "file_read", {"file_path": "/tmp/a.py"})
        assert hit is not None
        assert hit.entry.capability_name == "file_read"

    def test_cache_miss(self):
        cache = SessionToolCache()
        hit = cache.get("nonexistent_key", "file_read", {"file_path": "/tmp/b.py"})
        assert hit is None
        assert cache.miss_count == 1

    def test_lru_eviction(self):
        cache = SessionToolCache(max_entries=50)
        # Fill beyond MAX_CACHE_ENTRIES
        for i in range(55):
            key = f"file_read:/tmp/file{i}.py:0:0:g0"
            cache.set(
                key,
                "file_read",
                {"file_path": f"/tmp/file{i}.py"},
                f"content {i}",
                step_number=i,
            )
        # Cache should not exceed 50 entries
        assert cache.entry_count <= 50

    def test_invalidate_file(self):
        cache = SessionToolCache()
        params = {"file_path": "/tmp/x.py"}
        key = cache.generate_key("file_read", params)
        assert key is not None
        cache.set(key, "file_read", params, "hello", step_number=1)
        assert cache.entry_count == 1

        cache.invalidate_file("/tmp/x.py")
        # Entry should be removed
        assert cache.entry_count == 0

    def test_invalidate_from_bash(self):
        cache = SessionToolCache()
        params = {"file_path": "/tmp/src/app.py"}
        key = cache.generate_key("file_read", params)
        assert key is not None
        cache.set(key, "file_read", params, "content", step_number=1)
        assert cache.entry_count == 1

        # A broad mutating command like 'make' invalidates all file entries
        cache.invalidate_from_bash("make build", success=True)
        assert cache.entry_count == 0

    def test_non_mutating_bash_no_invalidation(self):
        cache = SessionToolCache()
        params = {"file_path": "/tmp/src/app.py"}
        key = cache.generate_key("file_read", params)
        assert key is not None
        cache.set(key, "file_read", params, "content", step_number=1)
        assert cache.entry_count == 1

        # Read-only bash commands should not invalidate
        cache.invalidate_from_bash("ls -la /tmp", success=True)
        assert cache.entry_count == 1

    def test_get_file_read_from_full_cache(self):
        cache = SessionToolCache()
        params = {"file_path": "/tmp/full.py"}
        key = cache.generate_key("file_read", params)
        assert key is not None
        cache.set(key, "file_read", params, "full file content", step_number=1)

        hit = cache.get_file_read_from_full_cache("/tmp/full.py")
        assert hit is not None

    def test_build_knowledge_inventory(self):
        cache = SessionToolCache()
        # Add a file read
        p1 = {"file_path": "/tmp/a.py"}
        k1 = cache.generate_key("file_read", p1)
        cache.set(k1, "file_read", p1, "aaa", step_number=1)

        # Add a search
        p2 = {"pattern": "TODO"}
        k2 = cache.generate_key("file_search", p2)
        cache.set(k2, "file_search", p2, "found stuff", step_number=2)

        inv = cache.build_knowledge_inventory()
        assert len(inv.files_read) == 1
        assert len(inv.searches_performed) == 1

    def test_partial_clear(self):
        cache = SessionToolCache()
        for i in range(20):
            key = f"file_read:/tmp/pc{i}.py:0:0:g0"
            cache.set(key, "file_read", {"file_path": f"/tmp/pc{i}.py"}, f"c{i}", step_number=i)
        assert cache.entry_count == 20

        cache.partial_clear(max_entries=5)
        assert cache.entry_count <= 5

    def test_extract_preview(self):
        cache = SessionToolCache()
        # Short output returns full text
        short = "line1\nline2"
        assert cache._extract_preview(short) == short

        # Long output returns head + tail with omission marker
        long_text = "\n".join(f"line {i}" for i in range(100))
        preview = cache._extract_preview(long_text)
        assert "omitted" in preview
        assert "line 0" in preview
        assert "line 99" in preview


class TestFormatKnowledgeInventory:
    def test_returns_none_when_empty(self):
        inv = KnowledgeInventory()
        assert format_knowledge_inventory(inv) is None

    def test_returns_formatted_string(self):
        inv = KnowledgeInventory(
            files_read=["/tmp/a.py"],
            searches_performed=["TODO"],
            hit_count=3,
            tokens_saved=100,
        )
        result = format_knowledge_inventory(inv)
        assert result is not None
        assert "Files read" in result
        assert "Searches" in result
        assert "Cache hits: 3" in result
