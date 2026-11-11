"""Keyword memory search on BM25 over character bigrams (rune.memory.lexical).

Whitespace word overlap, which memory used before, misses Korean whenever the particle differs
(커피를 / 커피는) and drops two-character words entirely in the durable-fact ranker. The retrieval
set pins what keyword search must find, per bucket, and what it can't (paraphrases need embeddings).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rune.memory import lexical
from rune.memory.lexical import LexicalIndex, terms
from rune.memory.manager import MemoryManager, _rank_durable_facts
from rune.memory.store import MemoryStore
from rune.memory.types import Episode, VectorMetadata
from rune.memory.vector import KeywordIndex
from tests.unit.memory_retrieval_set import DOCS, QUERIES

IDS = list(DOCS)
TEXTS = [DOCS[i] for i in IDS]


def recall_at(k: int, bucket: str | None = None, ranker=lexical.rank) -> float:
    """Share of queries whose memory is in the top k; ties are broken against it."""
    picked = [(q, rel) for b, q, rel in QUERIES if bucket in (None, b)]
    hits = 0
    for query, relevant in picked:
        scores = ranker(query, TEXTS)
        mine = scores[IDS.index(relevant)]
        ahead = sum(1 for i, s in enumerate(scores) if s >= mine and IDS[i] != relevant)
        hits += mine > 0 and ahead < k
    return hits / len(picked)


def whitespace_overlap(query: str, texts: list[str]) -> list[float]:  # the ranker this replaced
    words = set(query.lower().split())
    return [len(words & set(t.lower().split())) / len(words) for t in texts]


class TestTerms:
    def test_particles_and_endings_share_the_stem(self):
        assert set(terms("커피는")) & set(terms("커피를")) == {"커피"}
        assert set(terms("databases")) >= set(terms("database"))

    def test_punctuation_splits_words_and_one_character_words_stay(self):
        assert terms("invoice-service: a") == ["in", "nv", "vo", "oi", "ic", "ce", "se", "er", "rv", "vi", "ic", "ce",
                                                "a"]
        assert terms("") == [] and terms("  ,  ") == []


class TestRetrievalSet:
    def test_keyword_search_finds_what_shares_a_stem_in_both_languages(self):
        assert recall_at(7, "lex_en") == 1.0
        assert recall_at(7, "lex_ko") == 1.0
        assert recall_at(7, "particle_ko") >= 0.9, "a different particle must not hide the memory"

    def test_it_beats_the_whitespace_overlap_it_replaced(self):
        assert recall_at(7) >= 0.6  # 0.65 measured, ties broken against the memory
        assert recall_at(7, "particle_ko") > recall_at(7, "particle_ko", whitespace_overlap) + 0.5

    def test_paraphrases_are_left_to_embeddings(self):
        # Documents what keyword search can't do, so a jump here means the set changed, not magic.
        assert recall_at(7, "cross") < 0.3


class TestLexicalIndex:
    def test_adding_an_id_again_replaces_it_and_removing_leaves_no_trace(self):
        index = LexicalIndex()
        index.add("a", "dark roast coffee")
        index.add("b", "kombucha")
        index.add("a", "oat milk")
        assert len(index) == 2 and index.scores("coffee") == {}
        index.remove("a")
        index.remove("missing")
        assert index.scores("oat milk") == {} and set(index.scores("kombucha")) == {"b"}
        index.remove("b")
        assert len(index) == 0 and index._postings == {} and index._total == 0

    def test_rarer_terms_weigh_more(self):
        index = LexicalIndex()
        for i in range(20):
            index.add(f"common{i}", "회의 일정 정리")
        index.add("rare", "한빛상사 회의")
        best, _ = index.search("한빛상사 회의는 언제야", 1)[0]
        assert best == "rare"

    def test_a_query_with_no_shared_term_scores_nothing(self):
        assert lexical.rank("zzz", ["coffee", "tea"]) == [0.0, 0.0]
        assert lexical.rank("anything", []) == []


class TestKeywordIndex:
    @staticmethod
    def meta(item_id: str, kind: str = "episode") -> VectorMetadata:
        return VectorMetadata(type=kind, id=item_id, summary=item_id)

    def test_same_id_replaces_cap_evicts_oldest_and_type_filter_holds(self):
        kw = KeywordIndex(max_items=2)
        kw.add("커피 원두 주문", self.meta("e1"))
        kw.add("커피 원두 주문 다시", self.meta("e1"))
        kw.add("커피 머신 청소", self.meta("e2", kind="fact"))
        assert len(kw) == 2
        kw.add("커피 필터 교체", self.meta("e3"))
        assert len(kw) == 2 and {r.id for r in kw.search("커피를", k=5)} == {"e2", "e3"}
        assert [r.id for r in kw.search("커피는", k=5, type_filter="episode")] == ["e3"]
        kw.remove("e3")
        assert [r.id for r in kw.search("커피", k=5)] == ["e2"]

    def test_absorbed_items_count_as_newest(self):
        # initialize builds the index off the loop, then absorbs what was saved meanwhile:
        # those are the newest, so the oldest refilled item is the one that goes.
        refilled = KeywordIndex(max_items=2)
        refilled.add("오래된 작업", self.meta("old"))
        refilled.add("중간 작업", self.meta("mid"))
        live = KeywordIndex()
        live.add("방금 저장한 작업", self.meta("new"))
        refilled.absorb(live)
        assert {r.id for r in refilled.search("작업", k=5)} == {"mid", "new"}

    def test_items_without_an_id_are_all_kept(self):
        kw = KeywordIndex()
        kw.add("alpha task", VectorMetadata(type="episode", id="", summary="a"))
        kw.add("alpha job", VectorMetadata(type="episode", id="", summary="b"))
        assert len(kw) == 2


def fact(category, key, value):
    from rune.memory.tiered_memory import DurableMemoryEntry

    return DurableMemoryEntry(category=category, key=key, value=value, confidence=1.0)


def test_durable_fact_ranking_sees_two_character_korean_words():
    # The old ranker dropped words of two characters or fewer, so it never saw 커피 at all.
    facts = [fact("preference", f"취미{i}", f"오래된 메모 {i}") for i in range(40)]
    facts.insert(0, fact("preference", "커피", "아메리카노를 좋아하고 시럽은 넣지 않음"))
    ranked = _rank_durable_facts(facts, goal="내가 커피는 어떻게 마시더라", max_facts=3)
    assert ranked[0].key == "커피", "the oldest fact, but the one the goal is about"


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "memory.db")
    yield s
    s.close()


def episode(summary: str, days_ago: float) -> Episode:
    when = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    return Episode(task_summary=summary, timestamp=when, result="done", importance=0.5)


async def test_a_restart_refills_keyword_search_from_sqlite(store):
    store.save_episode(episode("연말정산 서류를 정리하고 의료비 공제 항목을 계산했다", 3))
    manager = MemoryManager(store=store, vector_store=MagicMock())
    await manager.initialize()
    assert [r.metadata.summary[:4] for r in manager.keywords.search("연말정산에서 의료비는", k=3)] == ["연말정산"]


async def test_an_old_episode_the_query_matches_is_scored_not_just_the_latest_fifty(store):
    store.save_episode(episode("오사카 2박 3일 여행 일정을 짜고 맛집을 정리했다", 200))
    for i in range(80):
        store.save_episode(episode(f"주간 업무 보고서 {i}번 정리", i / 10))
    vectors = MagicMock()
    vectors.search.return_value = []
    manager = MemoryManager(store=store, vector_store=vectors)
    await manager.initialize()
    scored = await manager.score_episodes("오사카에서 갔던 맛집들", limit=7)
    assert scored[0]["episode"].task_summary.startswith("오사카")
    assert scored[0]["breakdown"]["vector"] == 1.0


async def test_memory_search_finds_a_fact_whatever_the_particle(store, monkeypatch):
    from rune.capabilities import memory_capability
    from rune.capabilities.memory_capability import MemorySearchParams, memory_search

    manager = MemoryManager(store=store, vector_store=MagicMock())
    await manager.initialize()
    manager.working.facts.update({"강아지": "이름은 보리이고 진돗개다", "출퇴근": "지하철로 한다"})
    monkeypatch.setattr("rune.memory.manager.get_memory_manager", lambda: manager)
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: store)
    monkeypatch.setattr(memory_capability, "_KEYWORD_FLOOR", 0.25)
    result = await memory_search(MemorySearchParams(query="우리 강아지는 무슨 종이야"))
    assert result.success and "보리" in result.output and "지하철" not in result.output
