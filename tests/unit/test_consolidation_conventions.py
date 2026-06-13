"""Auto-capture of project conventions during consolidation.

Consolidation captured task-outcome facts but not durable project conventions, so
an under-specified later task could not recall how the project does things. The
extraction now also pulls `conventions` and saves them under the "project"
category, which build_memory_context injects as Durable Knowledge.
"""

from __future__ import annotations

import pytest

from rune.memory.consolidation import _parse_extraction


def test_parse_extraction_includes_conventions():
    raw = (
        '{"commitments": [], "lessons": ["x"], "entities": [], "decisions": [], '
        '"conventions": ["every function needs a docstring", "validate inputs, raise ValueError"]}'
    )
    out = _parse_extraction(raw)
    assert out["conventions"] == [
        "every function needs a docstring",
        "validate inputs, raise ValueError",
    ]


def test_parse_extraction_conventions_default_empty_when_absent():
    # Older/partial LLM output without the key must not crash and default to [].
    out = _parse_extraction('{"lessons": ["a"]}')
    assert out["conventions"] == []


def test_parse_extraction_malformed_returns_conventions_key():
    out = _parse_extraction("not json at all")
    assert out["conventions"] == []  # fallback dict carries the key


@pytest.mark.asyncio
async def test_conventions_saved_under_project_category(monkeypatch):
    """End-to-end: extracted conventions are persisted as "project" facts at
    durable confidence (>=0.7), so build_memory_context auto-surfaces them."""
    import rune.memory.consolidation as consol

    # Fake episode + store + manager
    class _Ep:
        id = "ep1"
        task_summary = "add a geometry helper"
        result = "done"
        lessons = ""
        entities = ""

    class _Store:
        conn = type("C", (), {"execute": lambda *a, **k: None})()

        def get_recent_episodes(self, limit=50):
            return [_Ep()]

    class _Mgr:
        store = _Store()

    import rune.memory.manager as mgr_mod
    monkeypatch.setattr(mgr_mod, "get_memory_manager", lambda: _Mgr())

    # Fake LLM returns conventions
    class _Client:
        async def completion(self, **kw):
            return {"choices": [{"message": {"content": (
                '{"commitments": [], "lessons": [], "entities": [], "decisions": [], '
                '"conventions": ["use snake_case for all functions"]}'
            )}}]}

    import rune.llm.client as client_mod
    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _Client())

    saved: list[tuple] = []
    import rune.memory.markdown_store as md_store
    monkeypatch.setattr(md_store, "save_learned_fact",
                        lambda cat, key, val, conf: saved.append((cat, key, val, conf)))
    import rune.memory.state as state_mod
    monkeypatch.setattr(state_mod, "is_suppressed", lambda *_a, **_k: False)

    await consol.consolidate_episode("ep1")

    proj = [s for s in saved if s[0] == "project"]
    assert proj, f"no project-category fact saved; got {saved}"
    cat, key, val, conf = proj[0]
    assert "snake_case" in val
    assert conf >= 0.7  # durable threshold → injected by build_memory_context
