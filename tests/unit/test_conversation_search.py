"""Conversation transcript search over the real ConversationStore `turns` table.

Recall what was actually said, ranked semantically (catches paraphrase a keyword
search misses), with embeddings cached by content hash so repeat searches are cheap.
"""

from __future__ import annotations

import pytest

from rune.capabilities.memory_capability import (
    ConversationSearchParams,
    conversation_search,
)
from rune.conversation.store import ConversationStore


@pytest.fixture
def store(tmp_dir, monkeypatch):
    db = tmp_dir / "conversations.db"
    s = ConversationStore(db)
    s._conn.execute(
        "INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
        "VALUES ('c1','u','DB choice','2026-06-01','2026-06-01')"
    )
    s._conn.execute(
        "INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
        "VALUES ('c2','u','Lunch','2026-06-02','2026-06-02')"
    )

    def add(cid, order, role, content, archived=0):
        s._conn.execute(
            "INSERT INTO turns (id,conversation_id,role,content,timestamp,created_order,archived) "
            "VALUES (?,?,?,?,?,?,?)",
            (f"{cid}t{order}", cid, role, content, "2026-06-01T00:00:00", order, archived),
        )

    add("c1", 0, "user", "Should we use Postgres or MySQL?")
    add("c1", 1, "assistant", "Go with PostgreSQL, better JSON support.")
    add("c2", 0, "user", "where for lunch today")
    s._conn.commit()
    # Capability resolves the DB via conversations_db_path → point it here.
    monkeypatch.setattr("rune.utils.paths.conversations_db_path", lambda: db)
    yield s


def test_fetch_searchable_turns_and_window(store):
    turns = store.fetch_searchable_turns()
    # Returns all real turns (no recency/keyword bound) with a content hash.
    assert len(turns) == 3 and all(t["content_hash"] for t in turns)
    hit = next(t for t in turns if t["conversation_id"] == "c1")
    win = store.get_turn_window("c1", hit["created_order"], window=2)
    assert win and all("role" in w and "content" in w for w in win)


def test_archived_and_empty_turns_excluded(store):
    store._conn.execute(
        "INSERT INTO turns (id,conversation_id,role,content,timestamp,created_order,archived) "
        "VALUES ('c1a','c1','user','archived secret','2026-06-01',5,1)"
    )
    store._conn.execute(
        "INSERT INTO turns (id,conversation_id,role,content,timestamp,created_order) "
        "VALUES ('c1e','c1','user','   ','2026-06-01',6)"
    )
    store._conn.commit()
    turns = store.fetch_searchable_turns()
    assert not any("archived" in t["content"] for t in turns)
    assert all(t["content"].strip() for t in turns)


def test_embedding_cache_roundtrip_by_content_hash(store):
    h = ConversationStore.content_hash("hello world")
    store.cache_embeddings({h: [0.1, 0.2, 0.3]})
    got = store.get_cached_embeddings([h, "missing"])
    assert got[h] == pytest.approx([0.1, 0.2, 0.3]) and "missing" not in got


@pytest.mark.asyncio
async def test_search_keyword_fallback(monkeypatch, store):
    import rune.memory.manager as mgr_mod

    monkeypatch.setattr(
        mgr_mod,
        "get_memory_manager",
        lambda: (_ for _ in ()).throw(RuntimeError("no embed")),
    )
    res = await conversation_search(ConversationSearchParams(query="postgres mysql"))
    assert res.success and "DB choice" in res.output and "PostgreSQL" in res.output


@pytest.mark.asyncio
async def test_search_semantic_ranks_paraphrase_and_caches(monkeypatch, store):
    class _Mgr:
        calls = 0

        async def embed_batch(self, texts):
            _Mgr.calls += 1
            out = []
            for t in texts:
                tl = t.lower()
                hit = any(k in tl for k in ("postgres", "mysql", "database", "json", "relational"))
                out.append([1.0, 0.0] if hit else [0.0, 1.0])
            return out

    import rune.memory.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_memory_manager", lambda: _Mgr())
    q = "which relational database did we pick"  # no postgres/mysql keyword
    res = await conversation_search(ConversationSearchParams(query=q))
    assert res.success and "DB choice" in res.output
    assert "Lunch" not in res.output.split("DB choice")[0]  # DB ranked first
    # Cache populated → a second search embeds only the (new) query, not the turns.
    n = store._conn.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0]
    assert n >= 3


@pytest.mark.asyncio
async def test_empty_query_returns_no_match(monkeypatch, store):
    for q in ("", "   ", "\t"):
        res = await conversation_search(ConversationSearchParams(query=q))
        assert res.success and "No past conversations" in res.output


def test_limit_and_window_bounds_rejected():
    from pydantic import ValidationError

    for bad in ({"limit": 0}, {"limit": -1}, {"limit": 999}, {"window": -1}, {"window": 99}):
        with pytest.raises(ValidationError):
            ConversationSearchParams(query="x", **bad)


@pytest.mark.asyncio
async def test_embedding_count_mismatch_falls_back(monkeypatch, store):
    class _BadMgr:
        async def embed_batch(self, texts):
            return [[1.0, 0.0]]  # wrong length → fall back, never drop/crash

    import rune.memory.manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "get_memory_manager", lambda: _BadMgr())
    res = await conversation_search(ConversationSearchParams(query="postgres mysql"))
    assert res.success and "PostgreSQL" in res.output


@pytest.mark.asyncio
async def test_embed_on_write_populates_cache(monkeypatch, tmp_dir):
    """save() embeds new turns so the first search never backfills."""
    import rune.llm.local_embedding as le

    class _Prov:
        async def embed(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    monkeypatch.setattr(le, "get_embedding_provider", lambda: _Prov())
    from rune.conversation.manager import ConversationManager

    db = tmp_dir / "conversations.db"
    mgr = ConversationManager(ConversationStore(db))
    conv = mgr.start_conversation(user_id="u")
    mgr.add_turn(conv.id, "user", "postgres vs mysql decision for analytics")
    await mgr.end_conversation(conv.id)  # save → embed-on-write
    st = ConversationStore(db)
    assert st._conn.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0] >= 1


@pytest.mark.asyncio
async def test_embed_on_write_is_best_effort(monkeypatch, tmp_dir):
    """An embedding failure must never break conversation persistence."""
    import rune.llm.local_embedding as le

    def _boom():
        raise RuntimeError("no embedding model")

    monkeypatch.setattr(le, "get_embedding_provider", _boom)
    from rune.conversation.manager import ConversationManager

    db = tmp_dir / "conversations.db"
    mgr = ConversationManager(ConversationStore(db))
    conv = mgr.start_conversation(user_id="u")
    mgr.add_turn(conv.id, "user", "hello there")
    await mgr.end_conversation(conv.id)  # must not raise
    st = ConversationStore(db)
    assert st._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
