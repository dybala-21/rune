"""Thread safety tests for LocalEmbeddingEngine.

These tests exercise the serialization guarantees WITHOUT loading a real
llama.cpp model — we stub the inner model with a mock that records
overlapping call count.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rune.llm.local_embedding import LocalEmbeddingEngine


class _MockModel:
    """Fake Llama model: sleeps briefly and tracks concurrent entries."""

    def __init__(self) -> None:
        self._active = 0
        self.max_concurrent = 0
        self.total_calls = 0
        self._counter_lock = threading.Lock()

    def embed(self, text):
        with self._counter_lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.total_calls += 1
        try:
            time.sleep(0.01)
            return [0.1, 0.2, 0.3]
        finally:
            with self._counter_lock:
                self._active -= 1


@pytest.fixture
def engine() -> LocalEmbeddingEngine:
    eng = LocalEmbeddingEngine()
    # Bypass real model loading.
    eng._model = _MockModel()
    eng._model_id = "nomic-embed-text"
    eng._initialized = True
    from rune.llm.local_embedding import EMBEDDING_MODELS
    eng._model_config = EMBEDDING_MODELS["nomic-embed-text"]
    return eng


class TestSyncLock:
    def test_sync_calls_never_overlap(self, engine):
        """Parallel threads calling embed_sync must never enter model.embed concurrently."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(engine.embed_sync, f"t{i}") for i in range(32)]
            for f in futures:
                f.result()
        assert engine._model.max_concurrent == 1
        assert engine._model.total_calls == 32


class TestAsyncLock:
    @pytest.mark.asyncio
    async def test_async_calls_never_overlap(self, engine):
        """Concurrent awaits on embed_single serialize through the async lock."""
        await asyncio.gather(*(engine.embed_single(f"t{i}") for i in range(16)))
        assert engine._model.max_concurrent == 1
        assert engine._model.total_calls == 16

    @pytest.mark.asyncio
    async def test_async_and_batch_serialize(self, engine):
        """Mixed single + batch concurrent calls still serialize."""
        await asyncio.gather(
            engine.embed_single("a"),
            engine.embed(["b", "c", "d"]),
            engine.embed_single("e"),
        )
        assert engine._model.max_concurrent == 1
        # 1 + 3 + 1 = 5 underlying embed() calls
        assert engine._model.total_calls == 5


class TestDispose:
    def test_dispose_shuts_down_executor(self, engine):
        engine.dispose()
        # Shutdown is non-blocking; executor should reject new tasks.
        with pytest.raises(RuntimeError):
            engine._executor.submit(lambda: None)
