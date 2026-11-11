"""Memory works without the vector and embedding extras, as on a hosted machine built without them.

The import checks run in a fresh interpreter with numpy, faiss and llama_cpp made unimportable,
since this test process may already have them loaded.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rune.api import run_memory
from rune.llm import local_embedding
from rune.llm.local_embedding import LocalEmbeddingEngine
from rune.utils.process_worker import WorkerUnavailable

WITHOUT_EXTRAS = """
import asyncio, json, sys
for name in ("numpy", "faiss", "llama_cpp"):
    sys.modules[name] = None  # import raises ModuleNotFoundError

from rune.agent.memory_bridge import save_agent_result_to_memory
from rune.api.handlers.health import memory_search_mode
from rune.capabilities.memory_capability import (MemorySaveParams, MemorySearchParams, memory_save,
                                                 memory_search)
from rune.memory.manager import get_memory_manager

async def main():
    saved = await memory_save(MemorySaveParams(content="dark roast, no sugar", type="preference", key="coffee"))
    found = await memory_search(MemorySearchParams(query="coffee"))
    manager = get_memory_manager()
    await save_agent_result_to_memory("rename the invoice exporter", {"output": "renamed it", "success": True},
                                      manager, conversation_id="c1")
    episodes = manager.store.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    context = await manager.build_memory_context("what coffee do I like")
    print(json.dumps({"saved": saved.success, "found": found.success, "found_text": str(found.output),
                      "episodes": episodes, "context": context, "search": memory_search_mode(),
                      "loaded": [m for m in ("numpy", "faiss", "llama_cpp") if sys.modules.get(m)]}))

asyncio.run(main())
"""


def test_memory_saves_and_recalls_without_vector_or_embedding_extras(tmp_path):
    env = {"RUNE_HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", WITHOUT_EXTRAS], env=env, capture_output=True, text=True,
                          timeout=120, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["loaded"] == []
    assert result["saved"] and result["found"]
    assert "dark roast" in result["found_text"]
    assert result["episodes"] == 1
    assert "dark roast" in result["context"]
    assert result["search"] == {"mode": "keyword", "reason": "the vector extra (numpy, faiss) is not installed"}
    # no embedding model is fetched when it could never be loaded
    assert not (tmp_path / "models").exists()


def test_no_model_download_without_llama_cpp(monkeypatch, tmp_path):
    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: False)
    monkeypatch.setattr(local_embedding, "_models_dir", lambda: tmp_path)
    fetched = []
    engine = LocalEmbeddingEngine(worker=SimpleNamespace(last_error=None, status={"state": "idle"}))
    monkeypatch.setattr(engine, "_download_model", lambda model_id: fetched.append(model_id))
    monkeypatch.setattr(engine, "_schedule_download", lambda: fetched.append("scheduled"))
    try:
        for download in (False, True):
            with pytest.raises(WorkerUnavailable, match="embedding` extra"):
                engine._call(["hello"], kind="document", timeout=1, download=download)
        assert fetched == []
        assert engine._worker.last_error == "embedding_extra_missing"
    finally:
        engine._executor.shutdown(wait=False)
        engine._downloads.shutdown(wait=False)


def test_health_says_keyword_when_the_embedding_extra_is_missing(monkeypatch):
    from rune.api.handlers.health import memory_search_mode

    pytest.importorskip("faiss")
    monkeypatch.setattr(local_embedding, "embeddings_installed", lambda: False)
    assert memory_search_mode() == {"mode": "keyword",
                                    "reason": "the embedding extra (llama-cpp-python) is not installed"}


class _Manager:
    def __init__(self, build):
        self.build = build

    async def initialize(self):
        pass

    async def build_memory_context(self, goal):
        return await self.build(goal)


async def test_web_memory_context_returns_what_memory_built(monkeypatch):
    async def build(goal):
        return f"## Durable Knowledge\n- likes: {goal}\n"

    monkeypatch.setattr("rune.memory.manager.get_memory_manager", lambda: _Manager(build))
    assert await run_memory.memory_context_for("tea") == "## Durable Knowledge\n- likes: tea\n"
    assert await run_memory.memory_context_for("   ") == ""


async def test_web_memory_context_never_blocks_or_breaks_the_run(monkeypatch):
    import asyncio

    async def slow(goal):
        await asyncio.sleep(10)

    async def broken(goal):
        raise RuntimeError("store is gone")

    monkeypatch.setattr(run_memory, "MEMORY_CONTEXT_TIMEOUT_S", 0.05)
    for build in (slow, broken):
        monkeypatch.setattr("rune.memory.manager.get_memory_manager", lambda b=build: _Manager(b))
        assert await run_memory.memory_context_for("anything") == ""
