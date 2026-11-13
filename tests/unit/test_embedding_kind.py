"""Query and document embeddings: the worker, the engine and every caller.

EmbeddingGemma reaches its measured recall (Ko-StrategyQA recall@10 0.868) only when a query and a
document each carry their own prompt. These pin who sends which kind, what the worker does with it,
and the engine's document cache and batch pieces.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import socket
import sys
import threading
import types
from types import SimpleNamespace
from typing import Any

import pytest

from rune.llm import embedding_worker
from rune.llm.embedding_models import (
    DEFAULT_MODEL,
    EMBEDDING_MODELS,
    EmbeddingVector,
)
from rune.llm.local_embedding import LocalEmbeddingEngine
from rune.utils.process_worker import receive_message, send_message

FP = "a" * 64
DIMS = EMBEDDING_MODELS[DEFAULT_MODEL].dimensions


@pytest.fixture(autouse=True)
def _local_embeddings_on(monkeypatch):
    """The engine and the reindex run only when local embeddings are turned on."""
    monkeypatch.setenv("RUNE_LOCAL_EMBEDDINGS", "1")


# The worker, with llama.cpp replaced by a recorder


def _default_params() -> SimpleNamespace:
    return SimpleNamespace(use_extra_bufts=True)


# Stands in for llama_cpp.llama_cpp, the ctypes module the model defaults come from.
native = SimpleNamespace(llama_model_default_params=_default_params)


class FakeLlama:
    made: list[dict[str, Any]] = []
    seen: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        # Llama() reads its model defaults from the ctypes module while it loads.
        kwargs["use_extra_bufts"] = native.llama_model_default_params().use_extra_bufts
        FakeLlama.made.append(kwargs)

    def embed(self, text: str, truncate: bool = False) -> list[float]:
        assert truncate, "input longer than n_ctx must be cut, not refused"
        FakeLlama.seen.append(text)
        return [float(len(text))] + [0.0] * (DIMS - 1)

    def close(self) -> None:
        pass


@pytest.fixture
def worker(tmp_path, monkeypatch):
    FakeLlama.made, FakeLlama.seen = [], []
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama, llama_cpp=native))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setitem(embedding_worker.EMBEDDING_MODELS, DEFAULT_MODEL,
                        dataclasses.replace(EMBEDDING_MODELS[DEFAULT_MODEL], sha256=digest))
    ours, theirs = socket.socketpair()
    thread = threading.Thread(target=embedding_worker.serve, args=(theirs,), daemon=True)
    thread.start()
    stat = model.stat()
    stamp = [stat.st_ino, stat.st_size, stat.st_mtime_ns]

    def ask(texts: list[str], kind: str, **extra: Any) -> dict[str, Any]:
        payload = {"model_id": DEFAULT_MODEL, "model_path": str(model), "model_stamp": stamp,
                   "texts": texts, "kind": kind, **extra}
        send_message(ours, {"id": "1", "payload": payload}, None)
        return receive_message(ours, None)

    yield ask, digest
    ours.close()
    thread.join(5)


def test_queries_and_documents_each_get_their_own_prompt(worker):
    ask, _ = worker
    query = ask(["우리 집 반려견 기억해?"], "query")["result"]
    document = ask(["보리라는 진돗개"], "document")["result"]
    assert FakeLlama.seen == ["task: search result | query: 우리 집 반려견 기억해?",
                              "title: none | text: 보리라는 진돗개"]
    assert query["fingerprint"] == document["fingerprint"], "one vector space for both kinds"


def test_the_model_loads_with_a_fixed_context_on_the_gpu_when_there_is_one(worker):
    ask, _ = worker
    ask(["x"], "document")
    assert FakeLlama.made[0]["n_ctx"] == 512, "n_ctx=0 takes the model's trained context"
    assert FakeLlama.made[0]["n_gpu_layers"] == -1


def test_the_weights_load_once_without_a_repacked_copy(worker):
    ask, _ = worker
    ask(["x"], "document")
    assert FakeLlama.made[0]["use_extra_bufts"] is False, "repacking copies every weight: +230 MB"
    assert native.llama_model_default_params().use_extra_bufts is True, "the defaults are put back"


def test_cpu_only_keeps_every_layer_on_the_cpu(worker):
    ask, _ = worker
    ask(["x"], "document", cpu_only=True)
    assert FakeLlama.made[0]["n_gpu_layers"] == 0


def test_the_fingerprint_names_the_instruction_version(worker):
    import importlib.metadata

    ask, digest = worker
    fingerprint = ask(["x"], "document")["result"]["fingerprint"]
    fields = [DEFAULT_MODEL, digest, 768, "gemma-retrieval-v1", importlib.metadata.version("llama-cpp-python")]
    assert fingerprint == hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def test_a_model_file_that_does_not_match_its_pin_is_refused(worker, monkeypatch):
    ask, _ = worker
    monkeypatch.setitem(embedding_worker.EMBEDDING_MODELS, DEFAULT_MODEL,
                        dataclasses.replace(EMBEDDING_MODELS[DEFAULT_MODEL], sha256="0" * 64))
    assert ask(["x"], "document") == {"id": "1", "error": "ValueError"}
    assert FakeLlama.made == []


# The engine, with the worker process replaced


class StubWorker:
    def __init__(self, fingerprint: str = FP) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fingerprint = fingerprint
        self.status = {"state": "ready"}
        self.last_error = None

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.requests.append(payload)
        dims = EMBEDDING_MODELS[payload["model_id"]].dimensions
        return {"dimensions": dims, "fingerprint": self.fingerprint,
                "vectors": [[float(len(t))] + [0.0] * (dims - 1) for t in payload["texts"]]}

    def invalidate(self, reason: str) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def engine(tmp_path, monkeypatch):
    (tmp_path / EMBEDDING_MODELS[DEFAULT_MODEL].file).write_bytes(b"fixture")
    monkeypatch.setattr("rune.llm.local_embedding._models_dir", lambda: tmp_path)
    monkeypatch.setattr("rune.llm.local_embedding.embeddings_installed", lambda: True)
    stub = StubWorker()
    eng = LocalEmbeddingEngine(worker=stub)  # type: ignore[arg-type]
    yield eng, stub
    eng.dispose()


def test_documents_embedded_before_come_from_the_cache_and_queries_never_do(engine):
    eng, stub = engine
    first = eng.embed_batch_sync(["rule one", "rule two"], kind="document")
    again = eng.embed_batch_sync(["rule two", "rule one", "rule three"], kind="document")
    assert [r["texts"] for r in stub.requests] == [["rule one", "rule two"], ["rule three"]]
    assert again[0] == first[1] and again[1] == first[0] and again[0].fingerprint == FP
    eng.embed_batch_sync(["rule one"], kind="query")
    eng.embed_batch_sync(["rule one"], kind="query")
    assert [r["kind"] for r in stub.requests[2:]] == ["query", "query"]


def test_a_new_model_drops_the_cache_instead_of_mixing_vectors(engine):
    eng, stub = engine
    eng.embed_batch_sync(["kept"], kind="document")
    stub.fingerprint = "b" * 64
    out = eng.embed_batch_sync(["kept", "new"], kind="document")
    assert {v.fingerprint for v in out} == {"b" * 64}
    assert stub.requests[-1]["texts"] == ["kept", "new"]


def test_the_cache_keeps_the_most_recently_used(engine, monkeypatch):
    monkeypatch.setattr("rune.llm.local_embedding._DOCUMENT_CACHE_SIZE", 2)
    eng, stub = engine
    eng.embed_batch_sync(["a", "b"], kind="document")
    eng.embed_batch_sync(["a"], kind="document")  # a is now newer than b
    eng.embed_batch_sync(["c"], kind="document")  # b goes
    stub.requests.clear()
    eng.embed_batch_sync(["a", "b", "c"], kind="document")
    assert stub.requests[0]["texts"] == ["b"]


async def test_a_large_document_batch_goes_in_pieces_each_under_the_character_budget(engine):
    eng, stub = engine
    turns = [f"{i:03d} " + "대화 내용 " * 80 for i in range(64)]  # ~500 characters each
    out = await eng.embed(turns, kind="document")
    assert [v[0] for v in out] == [float(len(t)) for t in turns], "order survives the pieces"
    assert len(stub.requests) > 1
    assert all(sum(map(len, r["texts"])) <= 4000 for r in stub.requests)


async def test_a_query_batch_is_one_call(engine):
    eng, stub = engine
    await eng.embed(["a", "b", "c"], kind="query")
    assert [r["texts"] for r in stub.requests] == [["a", "b", "c"]]


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.mark.parametrize("body, ok", [(b"real weights", True), (b"tampered", False)])
def test_a_download_is_kept_only_when_it_matches_the_pinned_hash(tmp_path, monkeypatch, body, ok):
    monkeypatch.setattr("rune.llm.local_embedding._models_dir", lambda: tmp_path)
    monkeypatch.setitem(EMBEDDING_MODELS, DEFAULT_MODEL, dataclasses.replace(
        EMBEDDING_MODELS[DEFAULT_MODEL], sha256=hashlib.sha256(b"real weights").hexdigest()))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: FakeResponse(body))
    eng = LocalEmbeddingEngine(worker=StubWorker())  # type: ignore[arg-type]
    try:
        if ok:
            assert eng._download_model(DEFAULT_MODEL).read_bytes() == body
        else:
            with pytest.raises(ValueError, match="pinned hash"):
                eng._download_model(DEFAULT_MODEL)
            assert not (tmp_path / EMBEDDING_MODELS[DEFAULT_MODEL].file).exists()
        assert not list(tmp_path.glob(".download-*")), "no partial file is left behind"
    finally:
        eng.dispose()


# Every caller sends the right kind


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fingerprint = FP

    def _vectors(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector([1.0, float(len(t))], FP) for t in texts]

    async def embed(self, texts: list[str], *, kind: str = "document") -> list[EmbeddingVector]:
        self.calls.append((kind, list(texts)))
        return self._vectors(texts)

    async def embed_single(self, text: str, *, kind: str = "document") -> EmbeddingVector:
        return (await self.embed([text], kind=kind))[0]

    def embed_sync(self, texts: list[str], *, kind: str = "document") -> list[EmbeddingVector]:
        self.calls.append((kind, list(texts)))
        return self._vectors(texts)


@pytest.fixture
def provider(monkeypatch):
    recorder = RecordingProvider()
    monkeypatch.setattr("rune.llm.local_embedding.get_embedding_provider", lambda: recorder)
    return recorder


async def test_hybrid_search_embeds_its_query_as_a_query(provider):
    from rune.memory.search import _retrieve_vectors

    vectors = SimpleNamespace(search=lambda embedding, k, type_filter: [])
    await _retrieve_vectors("커피 취향", vectors, 5, None)  # type: ignore[arg-type]
    assert provider.calls == [("query", ["커피 취향"])]


def test_skill_matching_embeds_the_goal_as_a_query_and_skills_as_documents(provider):
    from rune.skills.matcher import _semantic_scores
    from rune.skills.types import Skill

    skills = [Skill(name="pdf", description="Fill in PDF forms"), Skill(name="git", description="Commit")]
    assert set(_semantic_scores("양식 채워줘", skills)) == {"pdf", "git"}
    assert [kind for kind, _ in provider.calls] == ["query", "document"]
    assert provider.calls[0][1] == ["양식 채워줘"]


async def test_rule_retrieval_embeds_the_goal_as_a_query_and_rules_as_documents(provider, monkeypatch):
    from rune.memory import rule_learner

    rules = [{"key": "A", "value": "run tests", "confidence": 0.9, "domain": "code", "hit_count": 0}]
    monkeypatch.setattr(rule_learner, "_resolved_rule_candidates", lambda: rules)
    await rule_learner.get_relevant_rules("테스트 고쳐줘", domain="")
    assert provider.calls == [("query", ["테스트 고쳐줘"]), ("document", ["A: run tests"])]


async def test_a_saved_episode_is_indexed_as_a_document_and_scoring_embeds_a_query(provider, tmp_path):
    from rune.memory.manager import MemoryManager
    from rune.memory.store import MemoryStore
    from rune.memory.types import Episode

    vectors = SimpleNamespace(add=lambda embedding, meta: None, search=lambda embedding, k: [])
    store = MemoryStore(db_path=tmp_path / "memory.db")
    manager = MemoryManager(store=store, vector_store=vectors)  # type: ignore[arg-type]
    try:
        await manager.save_episode(Episode(task_summary="오사카 여행 일정", timestamp="2026-11-01T00:00:00+00:00"))
        await manager.score_episodes("오사카에서 갔던 곳", limit=3)
    finally:
        store.close()
    assert provider.calls == [("document", ["오사카 여행 일정"]), ("query", ["오사카에서 갔던 곳"])]


async def test_transcript_search_embeds_the_query_as_a_query_and_only_a_budget_of_turns(provider, monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    from rune.capabilities import memory_capability
    from rune.capabilities.memory_capability import ConversationSearchParams, conversation_search
    from rune.conversation.store import ConversationStore
    from rune.memory.manager import MemoryManager

    db = tmp_path / "conversations.db"
    store = ConversationStore(db)
    store._conn.execute("INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
                        "VALUES ('c','u','t','2026-06-01','2026-06-01')")
    for i in range(30):
        store._conn.execute(
            "INSERT INTO turns (id,conversation_id,role,content,timestamp,created_order) VALUES (?,?,?,?,?,?)",
            (f"t{i}", "c", "user", f"{i:02d} " + "말 " * 200, f"2026-06-01T00:00:{i:02d}", i))
    store._conn.commit()
    monkeypatch.setattr("rune.utils.paths.conversations_db_path", lambda: db)
    monkeypatch.setattr(memory_capability, "_INLINE_TURN_CHARS", 1500)
    manager = MemoryManager(store=SimpleNamespace(), vector_store=SimpleNamespace())  # type: ignore[arg-type]
    monkeypatch.setattr("rune.memory.manager.get_memory_manager", lambda: manager)
    await conversation_search(ConversationSearchParams(query="그때 했던 말"))
    assert provider.calls[0] == ("query", ["그때 했던 말"])
    documents = [t for kind, texts in provider.calls[1:] for t in texts if kind == "document"]
    assert {kind for kind, _ in provider.calls[1:]} == {"document"}
    assert sum(map(len, documents)) <= 1500 and documents, "the newest turns, within the budget"
    assert documents[0].startswith("29"), "recent first"


# The background reindex


async def test_reindex_fills_missing_episode_vectors_and_turns_once(provider, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    from rune.conversation.store import ConversationStore
    from rune.memory import reindex
    from rune.memory.store import MemoryStore
    from rune.memory.types import Episode, VectorMetadata
    from rune.memory.vector import VectorStore

    store = MemoryStore(db_path=tmp_path / "memory.db")
    for i in range(20):
        store.save_episode(Episode(id=f"e{i}", task_summary=f"작업 {i} " * (i + 1),
                                   timestamp=f"2026-11-01T00:00:{i:02d}+00:00"))
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: store)
    vectors = VectorStore(index_path=tmp_path / "vectors", require_identity=True)
    vectors.add(EmbeddingVector([1.0, 1.0], FP), VectorMetadata(type="episode", id="e3", summary="already"))
    db = tmp_path / "conversations.db"
    turns = ConversationStore(db)
    turns._conn.execute("INSERT INTO conversations (id,user_id,title,created_at,updated_at) "
                        "VALUES ('c','u','t','2026-06-01','2026-06-01')")
    for i in range(5):
        turns._conn.execute(
            "INSERT INTO turns (id,conversation_id,role,content,timestamp,created_order) VALUES (?,?,?,?,?,?)",
            (f"t{i}", "c", "user", f"turn {i}", f"2026-06-01T00:00:0{i}", i))
    turns._conn.commit()
    monkeypatch.setattr("rune.utils.paths.conversations_db_path", lambda: db)
    monkeypatch.setattr("rune.memory.markdown_indexer.collect_all_chunks", lambda: [])
    try:
        assert await reindex._episodes(provider, vectors) == 19
        assert vectors.indexed_ids({"episode"}) == {f"e{i}" for i in range(20)}
        pieces = [texts for kind, texts in provider.calls[1:]]
        assert all(len(p) <= 8 and sum(map(len, p)) <= 2000 for p in pieces)
        assert await reindex._turns(provider) == 5
        assert await reindex._turns(provider) == 0, "turns cached once are not embedded again"
        provider.calls.clear()
        assert await reindex._episodes(provider, vectors) == 0
        assert len(provider.calls) == 1, "only the probe that names the model"
        reloaded = VectorStore(index_path=tmp_path / "vectors", require_identity=True)
        reloaded.select_model(FP, 2)
        assert len(reloaded.indexed_ids({"episode"})) == 20, "the index was saved"
    finally:
        store.close()


async def test_keep_vectors_filled_waits_for_the_model_and_retries_a_failure(monkeypatch):
    from rune.memory import reindex

    monkeypatch.setattr("rune.llm.local_embedding.embeddings_installed", lambda: True)
    pytest.importorskip("faiss")
    ready = iter([False, True, True])
    attempts = []

    async def fill() -> dict[str, int]:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("worker down")
        return {"episodes": 1}

    await asyncio.wait_for(reindex.keep_vectors_filled(fill=fill, ready=lambda: next(ready), wait=0), 5)
    assert len(attempts) == 2


async def test_keep_vectors_filled_does_nothing_without_the_embedding_extra(monkeypatch):
    from rune.memory import reindex

    monkeypatch.setattr("rune.llm.local_embedding.embeddings_installed", lambda: False)

    async def fill() -> dict[str, int]:
        raise AssertionError("must not run")

    await asyncio.wait_for(reindex.keep_vectors_filled(fill=fill, ready=lambda: True, wait=0), 1)


async def test_markdown_chunks_sharing_an_id_are_embedded_once_and_not_again(provider, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    import rune.memory.markdown_indexer as indexer
    from rune.memory.vector import VectorStore

    chunks = [{"id": "fact:b", "text": "커피는 블랙", "hash": "h3", "type": "md_fact"},
              {"id": "fact:a", "text": "Result: completed", "hash": "h1", "type": "md_fact"},
              {"id": "fact:a", "text": "Result: completed again", "hash": "h2", "type": "md_fact"}]
    monkeypatch.setattr(indexer, "collect_all_chunks", lambda: list(chunks))
    vectors = VectorStore(index_path=tmp_path / "vectors", require_identity=True)
    first = await indexer.incremental_reindex(vectors)
    assert first["added"] == 2
    assert [r.text for r in vectors.search(EmbeddingVector([1.0, 23.0], FP), k=1)] == ["Result: completed again"]
    provider.calls.clear()
    assert (await indexer.incremental_reindex(vectors))["unchanged"] == 2
    assert [kind for kind, _ in provider.calls] == ["document"], "only the probe"
