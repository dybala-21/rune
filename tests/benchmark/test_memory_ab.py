"""A/B benchmark: Memory v1 (SQLite weighted-sum) vs v2 (Markdown RRF).

Measures:
  1. Write speed (fact save, daily log)
  2. Read speed (initialize, parse)
  3. Search quality (precision, recall on golden queries)
  4. Search latency
  5. End-to-end pipeline
"""

from __future__ import annotations

import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rune.memory.types import SearchResult, VectorMetadata
from rune.memory.vector import KeywordIndex

GOLDEN_FACTS = [
    ("preference", "package_manager", "pnpm"),
    ("preference", "test_runner", "vitest"),
    ("preference", "editor", "neovim"),
    ("preference", "indent", "2 spaces"),
    ("environment", "framework", "fastapi"),
    ("environment", "database", "postgresql 16"),
    ("environment", "python_version", "3.13"),
    ("environment", "deploy", "docker on aws ecs"),
    ("environment", "orm", "sqlalchemy 2.0"),
    ("lesson", "debugging", "check timezone first in datetime bugs"),
    ("lesson", "testing", "run tests after each file change"),
    ("lesson", "api", "always version your API endpoints"),
    ("decision", "auth", "switched from session to JWT"),
    ("decision", "database_choice", "postgresql over mysql for jsonb"),
    ("preference", "shell", "zsh"),
    ("preference", "terminal", "wezterm"),
    ("environment", "ci", "github actions"),
    ("environment", "container", "docker"),
    ("lesson", "performance", "profile before optimizing"),
    ("decision", "memory", "markdown over sqlite for user-facing facts"),
]

GOLDEN_QUERIES = [
    ("what package manager", "package_manager", "pnpm"),
    ("database", "database", "postgresql 16"),
    ("auth", "auth", "switched from session to JWT"),
    ("testing", "testing", "run tests after each file change"),
    ("timezone bug", "debugging", "check timezone first in datetime bugs"),
    ("python version", "python_version", "3.13"),
    ("editor preference", "editor", "neovim"),
    ("deployment", "deploy", "docker on aws ecs"),
    ("orm framework", "orm", "sqlalchemy 2.0"),
    ("api design", "api", "always version your API endpoints"),
]

DAILY_ENTRIES = [
    ("Fix auth bug in gateway", ["read auth.py", "edited token check", "tests passed"]),
    ("Implement rate limiting", ["added middleware", "configured redis", "load tested"]),
    ("Refactor database layer", ["extracted repository pattern", "updated 12 files"]),
    ("Debug timezone issue", ["found UTC mismatch", "fixed datetime.now()", "added regression test"]),
    ("Setup CI pipeline", ["created github actions workflow", "added test matrix"]),
    ("Write API documentation", ["generated openapi spec", "added examples"]),
    ("Optimize search performance", ["profiled hot path", "added index on timestamp"]),
    ("Fix CORS configuration", ["updated allowed origins", "tested cross-origin"]),
    ("Add user profile endpoint", ["created schema", "implemented CRUD", "wrote tests"]),
    ("Migrate to pnpm", ["removed yarn.lock", "created pnpm-lock.yaml", "updated CI"]),
]


@pytest.fixture
def mem_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rune.memory.markdown_store.memory_dir", lambda: tmp_path)
    monkeypatch.setattr("rune.memory.state._state_dir", lambda: tmp_path / ".state")
    (tmp_path / ".state").mkdir()
    for f in ("fact-meta.json", "suppressed.json", "index-state.json"):
        (tmp_path / ".state" / f).write_text("{}", encoding="utf-8")
    (tmp_path / ".state" / "conflicts.json").write_text("[]", encoding="utf-8")
    (tmp_path / "daily").mkdir()
    return tmp_path


def _seed_v2(mem_dir: Path) -> None:
    """Seed markdown with golden facts."""
    from rune.memory.markdown_store import append_daily_entry, save_learned_fact

    # Facts to MEMORY.md (user-curated) and learned.md (auto-extracted)
    memory_md = mem_dir / "MEMORY.md"
    memory_lines = ["# Preferences\n"]
    learned_facts = []

    for cat, key, val in GOLDEN_FACTS:
        if cat == "preference":
            memory_lines.append(f"- {key}: {val}\n")
        else:
            learned_facts.append((cat, key, val))

    memory_lines.append("\n# Environment\n")
    memory_md.write_text("".join(memory_lines), encoding="utf-8")

    for cat, key, val in learned_facts:
        save_learned_fact(cat, key, val, 0.85, mem_dir / "learned.md")

    # Daily logs
    for i, (title, actions) in enumerate(DAILY_ENTRIES):
        append_daily_entry(
            title=title, actions=actions,
            date=f"2026-03-{10 + i:02d}", time_str=f"{9 + i}:00",
        )


class TestWriteSpeed:
    """Measure markdown write latency."""

    def test_v2_markdown_write(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import save_learned_fact

        times = []
        for cat, key, val in GOLDEN_FACTS:
            t0 = time.perf_counter()
            save_learned_fact(cat, key, val, 0.85, mem_dir / "learned.md")
            times.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        print(f"\n[v2 Markdown] fact write: avg={avg:.2f}ms p99={p99:.2f}ms")
        assert avg < 50

    def test_v2_daily_log_write(self, mem_dir: Path) -> None:
        from rune.memory.markdown_store import append_daily_entry

        times = []
        for i, (title, actions) in enumerate(DAILY_ENTRIES):
            t0 = time.perf_counter()
            append_daily_entry(
                title=title, actions=actions,
                date="2026-03-16", time_str=f"{9 + i}:00",
            )
            times.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        print(f"\n[v2 Markdown] daily write: avg={avg:.2f}ms p99={p99:.2f}ms")
        assert avg < 50


class TestReadSpeed:
    """Measure markdown read/parse latency."""

    def test_v2_markdown_read(self, mem_dir: Path) -> None:
        _seed_v2(mem_dir)
        from rune.memory.markdown_store import parse_learned_md, parse_memory_md

        times_mem = []
        times_learned = []
        for _ in range(100):
            t0 = time.perf_counter()
            sections = parse_memory_md(mem_dir / "MEMORY.md")
            times_mem.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            facts = parse_learned_md(mem_dir / "learned.md")
            times_learned.append((time.perf_counter() - t0) * 1000)

        avg_mem = statistics.mean(times_mem)
        avg_learned = statistics.mean(times_learned)
        total_facts = sum(len(v) for v in sections.values()) + len(facts)
        print(f"\n[v2 Markdown] MEMORY.md parse: avg={avg_mem:.3f}ms")
        print(f"[v2 Markdown] learned.md parse: avg={avg_learned:.3f}ms")
        print(f"[v2 Markdown] total facts: {total_facts}")
        assert avg_mem < 5
        assert avg_learned < 5


class TestSearchQuality:
    """Compare search accuracy: weighted-sum vs RRF on identical data."""

    def _build_keyword_index(self, facts: list[tuple[str, str, str]]) -> KeywordIndex:
        from rune.memory.vector import KeywordIndex
        kw = KeywordIndex()
        for cat, key, val in facts:
            kw.add(
                f"{key}: {val}",
                VectorMetadata(type="fact", id=f"{cat}:{key}", summary=f"{key}: {val}"),
            )
        return kw

    def _search_weighted_sum(
        self, query: str, kw: KeywordIndex, k: int = 5,
    ) -> list[SearchResult]:
        """Old v1 algorithm: keyword-only with term overlap."""
        return kw.search(query, k=k)

    def _search_rrf(
        self, query: str, kw: KeywordIndex, k: int = 5,
    ) -> list[SearchResult]:
        """New v2 algorithm: RRF fusion (keyword only, no vectors in this test)."""
        from rune.memory.search import _mmr_select, _rrf_fuse

        kw_results = kw.search(query, k=k * 4)
        if not kw_results:
            return []
        # Single-source RRF (degenerates to rank-based scoring)
        fused = _rrf_fuse(kw_results)
        selected = _mmr_select(fused, k=k, lambda_=0.7)
        return selected

    def test_precision_comparison(self) -> None:
        kw = self._build_keyword_index(GOLDEN_FACTS)

        v1_hits = 0
        v2_hits = 0
        total = len(GOLDEN_QUERIES)

        print("\n--- Search Quality A/B ---")
        print(f"{'Query':<25} {'Expected':<20} {'v1 Hit':<8} {'v2 Hit':<8}")
        print("-" * 65)

        for query, expected_key, _expected_val in GOLDEN_QUERIES:
            v1_results = self._search_weighted_sum(query, kw, k=3)
            v2_results = self._search_rrf(query, kw, k=3)

            v1_found = any(expected_key in r.id for r in v1_results)
            v2_found = any(expected_key in r.id for r in v2_results)

            if v1_found:
                v1_hits += 1
            if v2_found:
                v2_hits += 1

            print(f"{query:<25} {expected_key:<20} {'Y' if v1_found else 'N':<8} {'Y' if v2_found else 'N':<8}")

        v1_precision = v1_hits / total * 100
        v2_precision = v2_hits / total * 100
        print(f"\nv1 precision: {v1_precision:.0f}% ({v1_hits}/{total})")
        print(f"v2 precision: {v2_precision:.0f}% ({v2_hits}/{total})")
        print(f"Delta: {v2_precision - v1_precision:+.0f}%")

        # v2 should be at least as good as v1
        assert v2_hits >= v1_hits, f"v2 ({v2_hits}) should not be worse than v1 ({v1_hits})"

    def test_search_latency(self) -> None:
        kw = self._build_keyword_index(GOLDEN_FACTS)

        v1_times = []
        v2_times = []

        for query, _, _ in GOLDEN_QUERIES:
            for _ in range(50):
                t0 = time.perf_counter()
                self._search_weighted_sum(query, kw, k=5)
                v1_times.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                self._search_rrf(query, kw, k=5)
                v2_times.append((time.perf_counter() - t0) * 1000)

        v1_avg = statistics.mean(v1_times)
        v2_avg = statistics.mean(v2_times)
        v1_p99 = sorted(v1_times)[int(len(v1_times) * 0.99)]
        v2_p99 = sorted(v2_times)[int(len(v2_times) * 0.99)]

        print(f"\n[v1] search latency: avg={v1_avg:.3f}ms p99={v1_p99:.3f}ms")
        print(f"[v2] search latency: avg={v2_avg:.3f}ms p99={v2_p99:.3f}ms")
        print(f"Delta: {v2_avg - v1_avg:+.3f}ms ({(v2_avg / v1_avg - 1) * 100:+.0f}%)")

        # v2 should be under 15ms total budget
        assert v2_avg < 15, f"v2 avg {v2_avg:.3f}ms exceeds 15ms budget"


class TestRRFStages:
    """Test each RRF pipeline stage independently."""

    def test_temporal_decay_correctness(self) -> None:
        from rune.memory.search import _apply_temporal_decay

        now = datetime.now(UTC)
        results = [
            SearchResult(id="today", score=1.0, metadata=VectorMetadata(
                type="md_daily", id="today", timestamp=now.isoformat())),
            SearchResult(id="week", score=1.0, metadata=VectorMetadata(
                type="md_daily", id="week",
                timestamp=(now.replace(day=max(1, now.day - 7))).isoformat())),
            SearchResult(id="month", score=1.0, metadata=VectorMetadata(
                type="md_daily", id="month",
                timestamp="2026-02-16T00:00:00+00:00")),
            SearchResult(id="evergreen", score=1.0, metadata=VectorMetadata(
                type="md_fact", id="evergreen", timestamp="2025-01-01T00:00:00+00:00")),
        ]

        _apply_temporal_decay(results)

        scores = {r.id: r.score for r in results}
        print("\nTemporal decay scores:")
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")

        assert scores["today"] > 0.95     # almost no decay
        assert scores["week"] > 0.80      # slight decay
        assert scores["month"] < 0.60     # moderate decay
        assert scores["evergreen"] == 1.0  # no decay (evergreen)
        assert scores["today"] > scores["week"] > scores["month"]

    def test_source_boost_ordering(self) -> None:
        from rune.memory.search import _apply_source_boost

        results = [
            SearchResult(id="ep", score=1.0, metadata=VectorMetadata(type="episode", id="ep")),
            SearchResult(id="fact", score=1.0, metadata=VectorMetadata(type="md_fact", id="fact")),
            SearchResult(id="proj", score=1.0, metadata=VectorMetadata(
                type="md_fact", id="proj", category="project")),
        ]

        _apply_source_boost(results)
        results.sort(key=lambda r: r.score, reverse=True)

        print("\nSource boost scores:")
        for r in results:
            print(f"  {r.id}: {r.score:.2f}")

        assert results[0].id == "proj"   # 1.3x
        assert results[1].id == "fact"   # 1.1x
        assert results[2].id == "ep"     # 0.9x

    def test_mmr_diversity(self) -> None:
        from rune.memory.search import _mmr_select

        # 3 near-identical + 1 diverse
        candidates = [
            SearchResult(id="a1", score=0.95, metadata=VectorMetadata(
                type="md_daily", id="a1", summary="fix auth bug in gateway api")),
            SearchResult(id="a2", score=0.90, metadata=VectorMetadata(
                type="md_daily", id="a2", summary="fix auth bug in gateway service")),
            SearchResult(id="a3", score=0.85, metadata=VectorMetadata(
                type="md_daily", id="a3", summary="fix auth bug in gateway module")),
            SearchResult(id="b1", score=0.80, metadata=VectorMetadata(
                type="md_fact", id="b1", summary="database: postgresql 16")),
        ]

        selected = _mmr_select(candidates, k=2, lambda_=0.7)
        ids = [r.id for r in selected]

        print(f"\nMMR selected: {ids}")
        # Should pick a1 (best) and b1 (diverse), not a1 + a2 (redundant)
        assert "a1" in ids
        assert "b1" in ids
        assert "a2" not in ids


class TestE2EPipeline:
    """End-to-end: seed data → search → verify results."""

    def test_full_v2_pipeline(self, mem_dir: Path) -> None:
        from rune.memory.markdown_indexer import collect_all_chunks
        from rune.memory.markdown_store import parse_learned_md, parse_memory_md

        # Seed
        _seed_v2(mem_dir)

        # Verify files created
        assert (mem_dir / "MEMORY.md").exists()
        assert (mem_dir / "learned.md").exists()
        daily_files = list((mem_dir / "daily").glob("*.md"))
        assert len(daily_files) == 10

        # Verify parse
        sections = parse_memory_md(mem_dir / "MEMORY.md")
        learned = parse_learned_md(mem_dir / "learned.md")
        total_facts = sum(len(v) for v in sections.values()) + len(learned)
        assert total_facts == len(GOLDEN_FACTS)

        # Verify chunks
        chunks = collect_all_chunks()
        assert len(chunks) >= 20  # facts + daily entries

        # Verify chunk types
        types = {c["type"] for c in chunks}
        assert "md_fact" in types
        assert "md_daily" in types

        print(f"\n[E2E] facts: {total_facts}, chunks: {len(chunks)}, daily files: {len(daily_files)}")

    def test_v2_write_then_read_consistency(self, mem_dir: Path) -> None:
        """Write facts, then read back — should be identical."""
        from rune.memory.markdown_store import parse_learned_md, save_learned_fact

        path = mem_dir / "learned.md"
        written = {}
        for cat, key, val in GOLDEN_FACTS:
            save_learned_fact(cat, key, val, 0.85, path)
            written[key] = val

        facts = parse_learned_md(path)
        read_back = {f["key"]: f["value"] for f in facts}

        missing = set(written) - set(read_back)
        extra = set(read_back) - set(written)
        mismatched = {k for k in written if k in read_back and written[k] != read_back[k]}

        print(f"\n[Consistency] written={len(written)} read={len(read_back)}")
        print(f"  missing: {missing or 'none'}")
        print(f"  extra: {extra or 'none'}")
        print(f"  mismatched: {mismatched or 'none'}")

        assert not missing, f"Facts lost: {missing}"
        assert not extra, f"Ghost facts: {extra}"
        assert not mismatched, f"Value mismatch: {mismatched}"
