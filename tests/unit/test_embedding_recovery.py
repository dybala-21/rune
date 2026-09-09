"""Memory revisions remain searchable and transient download errors recover."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rune.llm.embedding_models import DEFAULT_MODEL, EMBEDDING_MODELS, EmbeddingVector
from rune.llm.local_embedding import LocalEmbeddingEngine
from rune.memory.search import hybrid_search
from rune.memory.types import VectorMetadata
from rune.memory.vector import KeywordIndex, VectorStore
from rune.utils.process_worker import WorkerUnavailable


@pytest.mark.parametrize("type_filter", [None, "md_fact"])
async def test_many_updates_remain_searchable_after_reload(tmp_path, monkeypatch, type_filter):
    pytest.importorskip("faiss")
    query = EmbeddingVector([1.0, 0.0], "a" * 64)

    async def embed(text):
        return query

    monkeypatch.setattr("rune.llm.local_embedding.get_embedding_provider",
                        lambda: SimpleNamespace(embed_single=embed))
    store = VectorStore(index_path=tmp_path, require_identity=True)
    for revision in range(160):
        store.upsert(EmbeddingVector([1.0, revision * .001], "a" * 64),
                     VectorMetadata(type="md_fact", id="fact", summary=f"revision {revision}"))
    store.save()
    assert store.count < 65
    reloaded = VectorStore(index_path=tmp_path, require_identity=True)
    result = await hybrid_search("semantic query", reloaded, KeywordIndex(), k=5, type_filter=type_filter)
    assert [r.text for r in result] == ["revision 159"]
    assert reloaded.indexed_ids() == {"fact"}


def test_search_refills_after_type_and_deletion_filters(tmp_path):
    pytest.importorskip("faiss")
    store = VectorStore(dim=2, index_path=tmp_path)
    for i in range(200):
        store.add([i * .01, 0], VectorMetadata(type="episode", id=str(i)))
    for i in range(3):
        store.add([3 + i * .01, 0], VectorMetadata(type="md_fact", id=f"live-{i}"))
    assert len(store.search([0, 0], k=3, type_filter="md_fact")) == 3
    for i in range(200):
        store.delete_by_id(str(i))
    assert [r.id for r in store.search([0, 0], k=3)] == ["live-0", "live-1", "live-2"]
    store.save()
    assert store.count == 3
    assert store.search([0, 0], k=0) == []
    for i in range(3):
        store.delete_by_id(f"live-{i}")
    assert not store.search([0, 0])


@pytest.fixture
def recovery_engine(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rune.llm.local_embedding.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("rune.llm.local_embedding._models_dir", lambda: tmp_path)
    calls = []

    def request(payload, **kwargs):
        calls.append(payload)
        dimensions = EMBEDDING_MODELS[payload["model_id"]].dimensions
        return {"dimensions": dimensions, "fingerprint": "a" * 64,
                "vectors": [[1.0] * dimensions for _ in payload["texts"]]}

    worker = SimpleNamespace(request=request, close=lambda: None, last_error="")
    engine = LocalEmbeddingEngine(worker=worker)
    yield engine, clock, tmp_path, calls
    engine.dispose()


@pytest.mark.embedding_engine
def test_transient_download_recovers_once_after_cooldown(recovery_engine, monkeypatch):
    engine, clock, root, calls = recovery_engine
    attempts = []

    def download(model):
        attempts.append(model)
        if len(attempts) == 1:
            raise ConnectionError("temporarily offline")
        path = root / EMBEDDING_MODELS[model].file
        path.write_bytes(b"cached-model")
        return path

    monkeypatch.setattr(engine, "_download_model", download)
    with pytest.raises(WorkerUnavailable):
        engine.embed_sync("first")
    engine._download.result(timeout=2)
    for _ in range(5):
        with pytest.raises(WorkerUnavailable):
            engine.embed_sync("still cooling down")
    assert attempts == [DEFAULT_MODEL]
    clock[0] += 5

    def request(_):
        try:
            return engine.embed_sync("next")
        except WorkerUnavailable:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(request, range(8)))
    engine._download.result(timeout=2)
    assert len(engine.embed_sync("recovered")) == 768
    assert attempts == [DEFAULT_MODEL, DEFAULT_MODEL] and calls


@pytest.mark.embedding_engine
def test_repeated_download_failures_back_off_and_dispose_stops_retry(recovery_engine, monkeypatch):
    engine, clock, _, _ = recovery_engine
    attempts = []

    def download(model):
        attempts.append(model)
        raise ConnectionError("offline")

    monkeypatch.setattr(engine, "_download_model", download)
    for delay in (5, 10, 20, 40, 60, 60):
        with pytest.raises(WorkerUnavailable):
            engine.embed_sync("query")
        engine._download.result(timeout=2)
        previous = len(attempts)
        clock[0] += delay - .1
        with pytest.raises(WorkerUnavailable):
            engine.embed_sync("too early")
        assert len(attempts) == previous
        clock[0] += .1
    engine.dispose()
    clock[0] += 600
    with pytest.raises(WorkerUnavailable, match="closed"):
        engine.embed_sync("after close")
    assert len(attempts) == 6
