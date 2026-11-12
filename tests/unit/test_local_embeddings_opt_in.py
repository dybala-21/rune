"""Local embeddings stay off until the user turns them on, even with the embedding extra installed."""

from types import SimpleNamespace

import pytest

from rune.llm import local_embedding
from rune.llm.local_embedding import LocalEmbeddingEngine
from rune.utils.process_worker import WorkerUnavailable


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.setattr(local_embedding, "_models_dir", lambda: tmp_path)
    requests = []
    worker = SimpleNamespace(last_error=None, status={"state": "idle"},
                             request=lambda payload, timeout: requests.append(payload))
    engine = LocalEmbeddingEngine(worker=worker)
    engine.fetched = []
    engine.requests = requests
    monkeypatch.setattr(engine, "_download_model", lambda model_id: engine.fetched.append(model_id))
    monkeypatch.setattr(engine, "_schedule_download", lambda: engine.fetched.append("scheduled"))
    yield engine
    engine._executor.shutdown(wait=False)
    engine._downloads.shutdown(wait=False)


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_off_by_default_downloads_and_starts_nothing(engine, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    else:
        monkeypatch.setenv("RUNE_LOCAL_EMBEDDINGS", value)
    for download in (False, True):
        with pytest.raises(WorkerUnavailable, match="RUNE_LOCAL_EMBEDDINGS=1"):
            engine._call(["hello"], timeout=1, download=download)
    assert engine.fetched == [] and engine.requests == []
    assert engine._worker.last_error == "local_embeddings_off"
    assert not local_embedding.embeddings_enabled()


def test_turned_on_it_fetches_the_model_as_before(engine, monkeypatch):
    monkeypatch.setenv("RUNE_LOCAL_EMBEDDINGS", "1")
    assert local_embedding.embeddings_enabled()
    with pytest.raises(WorkerUnavailable, match="not cached"):
        engine._call(["hello"], timeout=1)
    assert engine.fetched == ["scheduled"]


def test_health_names_the_switch(monkeypatch):
    pytest.importorskip("faiss")
    from rune.api.handlers.health import memory_search_mode

    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    assert memory_search_mode() == {
        "mode": "keyword", "reason": "local embeddings are off (set RUNE_LOCAL_EMBEDDINGS=1 to turn them on)"}


async def test_warm_up_starts_no_worker_while_off(monkeypatch):
    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.setattr(local_embedding, "_is_model_cached", lambda: True)
    monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    started = []
    monkeypatch.setattr(local_embedding, "get_embedding_engine", lambda: started.append(1))
    await local_embedding.warm_up_embedding()
    assert started == []


async def test_saved_episode_is_not_embedded_while_off(monkeypatch):
    from rune.memory.manager import MemoryManager
    from rune.memory.types import Episode

    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    embedded, vectors = [], []

    async def embed_single(text):
        embedded.append(text)
        return [0.0]

    monkeypatch.setattr(local_embedding, "get_embedding_provider", lambda: SimpleNamespace(embed_single=embed_single))
    store = SimpleNamespace(save_episode=lambda episode: None)
    manager = MemoryManager(store=store, vector_store=SimpleNamespace(add=lambda v, m: vectors.append(m)))
    manager._tiered = SimpleNamespace(add_session_memory=lambda *a: None)
    await manager.save_episode(Episode(id="e1", task_summary="보리라는 진돗개를 키운다"))
    assert embedded == [] and vectors == []
    assert [r.id for r in manager.keywords.search("진돗개", k=5)] == ["e1"]


@pytest.mark.parametrize("on", [False, True])
async def test_search_asks_for_a_query_vector_only_when_on(monkeypatch, on):
    from rune.memory.search import hybrid_search
    from rune.memory.types import VectorMetadata
    from rune.memory.vector import KeywordIndex

    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    if on:
        monkeypatch.setenv("RUNE_LOCAL_EMBEDDINGS", "1")
    else:
        monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    asked = []

    async def embed_single(text):
        asked.append(text)
        raise WorkerUnavailable("not loaded")

    monkeypatch.setattr(local_embedding, "get_embedding_provider", lambda: SimpleNamespace(embed_single=embed_single))
    keywords = KeywordIndex()
    keywords.add("보리라는 진돗개를 키운다", VectorMetadata(type="episode", id="e1", summary=""))
    found = await hybrid_search("진돗개", SimpleNamespace(), keywords, k=3)
    assert [r.id for r in found] == ["e1"]
    assert asked == (["진돗개"] if on else [])


async def test_markdown_index_embeds_nothing_while_off(monkeypatch):
    from rune.memory import markdown_indexer

    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)
    monkeypatch.setattr(markdown_indexer, "collect_all_chunks",
                        lambda: (_ for _ in ()).throw(AssertionError("read chunks while off")))
    monkeypatch.setattr(local_embedding, "get_embedding_provider",
                        lambda: (_ for _ in ()).throw(AssertionError("asked for a provider while off")))
    stats = await markdown_indexer.incremental_reindex(SimpleNamespace())
    assert stats == {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}
