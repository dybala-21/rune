"""EmbeddingGemma as the local model: its catalog entry, the dense-weighted fusion, the per-model
match threshold and the worker that is stopped while idle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rune.llm import local_embedding
from rune.llm.embedding_models import DEFAULT_MODEL, EMBEDDING_MODELS
from rune.memory.types import SearchResult, VectorMetadata


def test_the_default_is_embeddinggemma_pinned_by_hash_with_its_retrieval_prompts():
    config = EMBEDDING_MODELS[DEFAULT_MODEL]
    assert DEFAULT_MODEL == "embeddinggemma-300m"
    assert (config.repo, config.file) == ("ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
                                          "embeddinggemma-300M-qat-Q4_0.gguf")
    assert config.sha256 == "50d28e22432a148f6f8a86eab3700f92add5d1f54baf7790675a2a4dadbccf26"
    assert (config.query_prefix, config.document_prefix) == ("task: search result | query: ",
                                                             "title: none | text: ")
    assert (config.dimensions, config.n_ctx, config.match_threshold) == (768, 512, 0.45)


# Fusion


def _hit(doc_id: str) -> SearchResult:
    return SearchResult(id=doc_id, score=1.0, metadata=VectorMetadata(type="md_fact", id=doc_id, summary=""),
                        text=doc_id)


def test_the_dense_list_outweighs_the_keyword_list_ten_to_one():
    from rune.memory.search import _rrf_fuse

    dense = [_hit("dense-1"), _hit("dense-2")]
    keyword = [_hit("keyword-1")]
    equal = [r.id for r in _rrf_fuse(dense, keyword)]
    weighted = [r.id for r in _rrf_fuse(dense, keyword, weights=(10.0, 1.0))]
    assert equal.index("keyword-1") < equal.index("dense-2"), "1:1: a keyword-only top hit passes dense #2"
    assert weighted == ["dense-1", "dense-2", "keyword-1"]
    with pytest.raises(ValueError):
        _rrf_fuse(dense, keyword, weights=(1.0,))


async def test_hybrid_search_fuses_with_the_dense_weight(monkeypatch):
    from rune.memory import search

    seen = {}
    original = search._rrf_fuse

    def spy(*lists, weights=None):
        seen["weights"] = weights
        return original(*lists, weights=weights)

    monkeypatch.setattr(search, "_rrf_fuse", spy)

    async def vectors(*_):
        return [_hit("a")]

    monkeypatch.setattr(search, "_retrieve_vectors", vectors)
    monkeypatch.setattr(search, "_retrieve_keywords", lambda *_: [_hit("b")])
    await search.hybrid_search("q", SimpleNamespace(), SimpleNamespace(), k=2)  # type: ignore[arg-type]
    assert seen["weights"] == (10.0, 1.0)


# Match threshold


def test_the_match_threshold_follows_the_active_model(monkeypatch):
    monkeypatch.setattr(local_embedding, "_engine", None)
    assert local_embedding.match_threshold() == 0.45
    monkeypatch.setattr(local_embedding, "_engine", SimpleNamespace(_model_id="nomic-embed-text"))
    assert local_embedding.match_threshold() == 0.55


class _Rules:
    _CANDS = [{"key": "near", "value": "v", "confidence": 0.9, "domain": "other", "hit_count": 0}]

    def __init__(self, cosine: float) -> None:
        # goal [1, 0] against a rule at the given cosine
        self.vecs = [[1.0, 0.0], [cosine, (1 - cosine**2) ** 0.5]]

    async def embed_batch(self, texts, kind="document"):
        return self.vecs[:1] if kind == "query" else self.vecs[1:]


@pytest.mark.parametrize("cosine, env, kept", [(0.5, None, True), (0.4, None, False), (0.5, "0.6", False)])
async def test_rules_match_from_the_model_threshold_unless_the_env_sets_one(monkeypatch, cosine, env, kept):
    import rune.memory.manager as manager
    from rune.memory import rule_learner

    monkeypatch.setattr(local_embedding, "_engine", None)
    if env is None:
        monkeypatch.delenv("RUNE_RULE_SIM_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("RUNE_RULE_SIM_THRESHOLD", env)
    monkeypatch.setattr(rule_learner, "_resolved_rule_candidates", lambda: list(_Rules._CANDS))
    monkeypatch.setattr(manager, "get_memory_manager", lambda: _Rules(cosine))
    found = await rule_learner.get_relevant_rules("goal", domain="")
    assert bool(found) is kept


@pytest.mark.parametrize("env, expected", [(None, 0.45), ("0.7", 0.7), ("junk", 0.45)])
def test_skills_match_from_the_model_threshold_unless_the_env_sets_one(monkeypatch, env, expected):
    from rune.skills.matcher import _sim_threshold

    monkeypatch.setattr(local_embedding, "_engine", None)
    if env is None:
        monkeypatch.delenv("RUNE_SKILL_SIM_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("RUNE_SKILL_SIM_THRESHOLD", env)
    assert _sim_threshold() == expected


# Idle worker


def test_a_parked_worker_still_counts_as_ready():
    engine = local_embedding.LocalEmbeddingEngine(
        worker=SimpleNamespace(status={"state": "parked"}, close=lambda: None))  # type: ignore[arg-type]
    try:
        engine._fingerprint = "a" * 64
        assert engine.is_ready() and engine.fingerprint == "a" * 64
        engine._worker.status = {"state": "degraded"}
        assert not engine.is_ready()
    finally:
        engine.dispose()


def test_the_engine_parks_its_worker_after_ten_idle_minutes():
    engine = local_embedding.LocalEmbeddingEngine()
    try:
        assert engine._worker._idle_timeout == 600.0
    finally:
        engine.dispose()


async def test_the_background_reindex_does_nothing_while_local_embeddings_are_off(monkeypatch):
    pytest.importorskip("faiss")
    from rune.memory import reindex

    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: True)
    monkeypatch.delenv("RUNE_LOCAL_EMBEDDINGS", raising=False)

    async def fill() -> dict[str, int]:
        raise AssertionError("must not run while off")

    await asyncio.wait_for(reindex.keep_vectors_filled(fill=fill, ready=lambda: True, wait=0), 1)
