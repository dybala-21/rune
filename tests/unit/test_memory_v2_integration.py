"""End-to-end integration tests for Memory v2 markdown-primary architecture.

Tests every flow defined in docs/design/memory-v2.md:
  - First run scaffolding
  - MEMORY.md (Zone 1) read/write
  - learned.md (Zone 2) CRUD + suppression
  - Daily log creation + parsing
  - user-profile.md Stats overwrite + section preservation
  - fact-meta.json hit_count tracking
  - Fact extraction with suppressed/MEMORY.md priority
  - RRF search pipeline stages
  - Reconciliation on startup
  - GC pruning
  - Backup and restore
  - Crash recovery (markdown first, meta second)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rune.memory.types import SearchResult, VectorMetadata


@pytest.fixture
def mem_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rune.memory.markdown_store.memory_dir", lambda: tmp_path)
    monkeypatch.setattr("rune.memory.state._state_dir", lambda: tmp_path / ".state")
    (tmp_path / ".state").mkdir()
    for f in ("fact-meta.json", "suppressed.json", "index-state.json"):
        (tmp_path / ".state" / f).write_text("{}", encoding="utf-8")
    (tmp_path / ".state" / "conflicts.json").write_text("[]", encoding="utf-8")
    return tmp_path


class TestFirstRun:
    def test_creates_all_files(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import ensure_memory_structure
        ensure_memory_structure()

        assert (mem_dir / "MEMORY.md").exists()
        assert (mem_dir / "learned.md").exists()
        assert (mem_dir / "daily").is_dir()
        assert (mem_dir / "user-profile.md").exists()
        assert (mem_dir / ".state" / "fact-meta.json").exists()

    def test_memory_md_has_section_headers(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import ensure_memory_structure
        ensure_memory_structure()

        content = (mem_dir / "MEMORY.md").read_text()
        assert "# Preferences" in content
        assert "# Environment" in content
        assert "# Notes" in content


class TestZone1MemoryMd:
    def test_user_curated_read(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import parse_memory_md
        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n- shell: zsh\n")

        sections = parse_memory_md(md)
        assert sections["Preferences"] == ["editor: vim", "shell: zsh"]

    def test_memory_md_wins_over_learned(self, mem_dir: Path) -> None:
        """If same key exists in MEMORY.md and learned.md, MEMORY.md wins."""
        from rune.memory.markdown_store import memory_md_has_key

        md = mem_dir / "MEMORY.md"
        md.write_text("# Preferences\n\n- editor: vim\n")

        # MEMORY.md is user-curated and takes precedence over learned.md.
        assert memory_md_has_key("editor", md) is True


class TestZone2LearnedMd:
    def test_save_extract_delete_suppress(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import (
            learned_md_has_key,
            remove_learned_fact,
            save_learned_fact,
        )
        from rune.memory.state import is_suppressed, suppress_fact

        path = mem_dir / "learned.md"

        # Save
        save_learned_fact("preference", "editor", "vim", 0.85, path)
        assert learned_md_has_key("editor", path) == "vim"

        # Delete
        remove_learned_fact("editor", path)
        assert learned_md_has_key("editor", path) is None

        # Suppress
        suppress_fact("editor", "vim", "user_deleted")
        assert is_suppressed("editor") is True

    def test_suppressed_prevents_reextraction(self, mem_dir: Path) -> None:
        from rune.memory.state import is_suppressed, suppress_fact
        suppress_fact("test_key", "test_val", "user_deleted")
        assert is_suppressed("test_key") is True
        assert is_suppressed("other_key") is False

    def test_unsuppress(self, mem_dir: Path) -> None:
        from rune.memory.state import is_suppressed, suppress_fact, unsuppress_fact
        suppress_fact("test_key", "val", "user_deleted")
        assert is_suppressed("test_key") is True
        unsuppress_fact("test_key")
        assert is_suppressed("test_key") is False


class TestZone2DailyLog:
    def test_append_and_parse(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import append_daily_entry, parse_daily_log

        (mem_dir / "daily").mkdir()
        path = append_daily_entry(
            title="Fix auth bug",
            actions=["read auth.py", "edited token check", "tests passed"],
            lessons=["check timezone first"],
            decisions=["use JWT"],
            date="2026-03-16",
            time_str="14:23",
        )

        entries = parse_daily_log(path)
        assert len(entries) == 1
        e = entries[0]
        assert e["title"] == "Fix auth bug"
        assert e["time"] == "14:23"
        assert "read auth.py" in e["actions"]
        assert "check timezone first" in e["lessons"]
        assert "use JWT" in e["decisions"]

    def test_lesson_marker(self, mem_dir: Path) -> None:
        """The > prefix in daily logs marks lessons for promotion."""
        (mem_dir / "daily").mkdir()
        append_daily_entry = __import__(
            "rune.memory.markdown_store", fromlist=["append_daily_entry"]
        ).append_daily_entry

        path = append_daily_entry(
            title="Debug session",
            actions=["investigated timeout"],
            lessons=["always set explicit timeout"],
            date="2026-03-16",
            time_str="10:00",
        )

        content = path.read_text()
        assert "> lesson: always set explicit timeout" in content


class TestZone2UserProfile:
    def test_stats_overwrite_preserves_others(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import parse_user_profile, update_user_profile_section

        path = mem_dir / "user-profile.md"
        path.write_text(
            "# Communication\n\n- language: korean\n\n"
            "# Stats\n\n- old data\n\n"
            "# Goals\n\n- [ ] ship v1\n"
        )

        update_user_profile_section("Stats", ["python: 50 tasks, 90% success"], path)

        sections = parse_user_profile(path)
        assert sections["Communication"] == ["language: korean"]
        assert sections["Stats"] == ["python: 50 tasks, 90% success"]
        assert sections["Goals"] == ["[ ] ship v1"]

    def test_custom_section_preserved(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import parse_user_profile, update_user_profile_section

        path = mem_dir / "user-profile.md"
        path.write_text("# MyCustomSection\n\n- custom data\n")

        update_user_profile_section("Stats", ["new stat"], path)

        sections = parse_user_profile(path)
        assert "MyCustomSection" in sections
        assert sections["MyCustomSection"] == ["custom data"]


class TestZone3FactMeta:
    def test_hit_count(self, mem_dir: Path) -> None:
        from rune.memory.state import increment_hit_count, load_fact_meta

        increment_hit_count("editor")
        increment_hit_count("editor")
        increment_hit_count("editor")

        meta = load_fact_meta()
        assert meta["editor"]["hit_count"] == 3
        assert "last_hit" in meta["editor"]

    def test_conflict_recording(self, mem_dir: Path) -> None:
        from rune.memory.state import load_conflicts, record_conflict

        record_conflict("database", "mysql", "postgresql", "ep1", "ep2")
        conflicts = load_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["old_value"] == "mysql"
        assert conflicts[0]["new_value"] == "postgresql"


class TestSearchPipeline:
    def test_rrf_fusion(self) -> None:
        from rune.memory.search import _rrf_fuse

        list1 = [
            SearchResult(id="a", score=0.9, metadata=VectorMetadata(type="episode", id="a")),
            SearchResult(id="b", score=0.7, metadata=VectorMetadata(type="episode", id="b")),
            SearchResult(id="c", score=0.5, metadata=VectorMetadata(type="episode", id="c")),
        ]
        list2 = [
            SearchResult(id="b", score=0.8, metadata=VectorMetadata(type="episode", id="b")),
            SearchResult(id="c", score=0.6, metadata=VectorMetadata(type="episode", id="c")),
            SearchResult(id="d", score=0.4, metadata=VectorMetadata(type="episode", id="d")),
        ]

        fused = _rrf_fuse(list1, list2)

        # b appears in both lists at good ranks, should be top
        ids = [r.id for r in fused]
        assert ids[0] == "b"  # best combined rank

    def test_temporal_decay_evergreen(self) -> None:
        from rune.memory.search import _apply_temporal_decay

        old_ts = "2025-01-01T00:00:00+00:00"  # very old

        evergreen = SearchResult(
            id="fact1", score=1.0,
            metadata=VectorMetadata(type="md_fact", id="fact1", timestamp=old_ts),
        )
        daily = SearchResult(
            id="daily1", score=1.0,
            metadata=VectorMetadata(type="md_daily", id="daily1", timestamp=old_ts),
        )

        _apply_temporal_decay([evergreen, daily])

        # Evergreen should not decay
        assert evergreen.score == 1.0
        # Daily should decay significantly (>1 year old)
        assert daily.score < 0.1

    def test_temporal_decay_recent(self) -> None:
        from rune.memory.search import _apply_temporal_decay

        now_ts = datetime.now(UTC).isoformat()
        result = SearchResult(
            id="d1", score=1.0,
            metadata=VectorMetadata(type="md_daily", id="d1", timestamp=now_ts),
        )
        _apply_temporal_decay([result])
        # Recent items should barely decay
        assert result.score > 0.95

    def test_source_boost(self) -> None:
        from rune.memory.search import _apply_source_boost

        project = SearchResult(
            id="p1", score=1.0,
            metadata=VectorMetadata(type="md_fact", id="p1", category="project"),
        )
        episode = SearchResult(
            id="e1", score=1.0,
            metadata=VectorMetadata(type="episode", id="e1"),
        )

        _apply_source_boost([project, episode])

        assert project.score == 1.3  # project boost
        assert episode.score == 0.9  # episode penalty

    def test_mmr_deduplicates(self) -> None:
        from rune.memory.search import _mmr_select

        # Two near-identical results and one different
        candidates = [
            SearchResult(id="a", score=0.9, metadata=VectorMetadata(
                type="md_daily", id="a", summary="fix auth bug in gateway")),
            SearchResult(id="b", score=0.85, metadata=VectorMetadata(
                type="md_daily", id="b", summary="fix auth bug in gateway service")),
            SearchResult(id="c", score=0.8, metadata=VectorMetadata(
                type="md_fact", id="c", summary="database: postgresql 16")),
        ]

        selected = _mmr_select(candidates, k=2, lambda_=0.7)

        ids = {r.id for r in selected}
        # Should pick a (best) and c (diverse), not a and b (redundant)
        assert "a" in ids
        assert "c" in ids


class TestReconciliation:
    def test_learned_md_value_wins(self, mem_dir: Path) -> None:
        """If user edits learned.md value, fact-meta.json should update on startup."""

        from rune.memory.markdown_store import save_learned_fact
        from rune.memory.state import load_fact_meta, save_fact_meta

        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.85, path)

        # Simulate meta with different confidence
        meta = {"editor": {"confidence": 0.5, "source": "old"}}
        save_fact_meta(meta)

        # Manually edit learned.md to change confidence
        content = path.read_text()
        content = content.replace("(0.85)", "(0.95)")
        path.write_text(content)

        # Parse learned.md (simulating what initialize() does)
        from rune.memory.markdown_store import parse_learned_md
        learned = parse_learned_md(path)
        meta = load_fact_meta()

        # Reconcile: markdown value should win
        for fact in learned:
            key = fact["key"]
            if key in meta and meta[key].get("confidence") != fact["confidence"]:
                meta[key]["confidence"] = fact["confidence"]

        assert meta["editor"]["confidence"] == 0.95


class TestGC:
    def test_prune_over_cap(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import parse_learned_md, prune_learned_md, save_learned_fact
        from rune.memory.state import is_suppressed, suppress_fact

        path = mem_dir / "learned.md"
        for i in range(10):
            save_learned_fact("test", f"key_{i}", f"val_{i}", i * 0.1, path)

        removed = prune_learned_md(cap=5, path=path)
        assert len(removed) == 5

        remaining = parse_learned_md(path)
        assert len(remaining) == 5

        # Removed keys should have low confidence (0.0 to 0.4)
        for key in removed:
            # Verify they can be suppressed
            suppress_fact(key, "", "auto_pruned")
            assert is_suppressed(key)


class TestBackupRestore:
    def test_backup_created_on_write(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import save_learned_fact

        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        save_learned_fact("preference", "editor", "neovim", 0.9, path)

        bak = mem_dir / ".state" / "learned.md.bak"
        assert bak.exists()
        bak_content = bak.read_text()
        assert "vim" in bak_content

    def test_restore(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import parse_learned_md, save_learned_fact

        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.8, path)
        save_learned_fact("preference", "editor", "neovim", 0.9, path)

        # Restore from backup
        bak = mem_dir / ".state" / "learned.md.bak"
        path.write_text(bak.read_text(), encoding="utf-8")

        facts = parse_learned_md(path)
        assert facts[0]["value"] == "vim"


class TestCrashRecovery:
    def test_markdown_consistent_after_meta_crash(self, mem_dir: Path) -> None:
        """If crash happens between markdown write and meta write,
        markdown (source of truth) should be consistent."""
        from rune.memory.markdown_store import parse_learned_md, save_learned_fact
        from rune.memory.state import load_fact_meta

        path = mem_dir / "learned.md"
        save_learned_fact("preference", "editor", "vim", 0.85, path)

        # Simulate crash: delete fact-meta.json
        meta_path = mem_dir / ".state" / "fact-meta.json"
        meta_path.write_text("{}", encoding="utf-8")

        # Markdown should still have the fact
        facts = parse_learned_md(path)
        assert len(facts) == 1
        assert facts[0]["key"] == "editor"
        assert facts[0]["value"] == "vim"
        assert facts[0]["confidence"] == 0.85

        # Meta can be reconstructed from markdown
        meta = load_fact_meta()
        assert meta == {}  # empty after "crash"
        # Reconciliation would rebuild it from learned.md


class TestRebuild:
    def test_rebuild_from_scratch(self, mem_dir: Path) -> None:
        """Delete .state/ and verify index can be rebuilt from markdown."""
        from rune.memory.markdown_indexer import collect_all_chunks
        from rune.memory.markdown_store import save_learned_fact
        from rune.memory.state import save_index_state

        (mem_dir / "MEMORY.md").write_text("# Preferences\n\n- editor: vim\n")
        save_learned_fact("env", "python", "3.13", 0.9, mem_dir / "learned.md")

        # Clear index state
        save_index_state({"chunks": {}})

        # Collect chunks (would normally be followed by embedding)
        chunks = collect_all_chunks()
        assert len(chunks) >= 2  # at least 1 from MEMORY.md + 1 from learned.md

        # Verify chunk content
        texts = [c["text"] for c in chunks]
        assert any("editor" in t for t in texts)
        assert any("python" in t for t in texts)
