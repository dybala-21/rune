"""Opt-in embedding-unavailable stub for harness regression tests.

Exercises fallback paths without loading llama.cpp. Native embeddings require
separate integration tests.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_native_embedding(monkeypatch):
    from rune.llm.local_embedding import LocalEmbeddingEngine

    def unavailable(*args, **kwargs):
        raise RuntimeError("Native embeddings excluded from verified-workflow regression run")

    monkeypatch.setattr(LocalEmbeddingEngine, "_activate_model", unavailable)
    monkeypatch.setattr(LocalEmbeddingEngine, "_download_model", unavailable)
