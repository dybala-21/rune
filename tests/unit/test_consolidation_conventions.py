"""Auto-capture of project conventions during consolidation.

Consolidation captured task-outcome facts but not durable project conventions, so
an under-specified later task could not recall how the project does things. The
extraction now also pulls `conventions` and saves them under the "project"
category, which build_memory_context injects as Durable Knowledge.
"""

from __future__ import annotations

import pytest

from rune.memory.consolidation import (
    _EXTRACTION_MAX_TOKENS,
    _is_mechanical_lesson,
    _parse_extraction,
)


def test_mechanical_lessons_detected_and_excluded():
    # RUNE's own save-time telemetry must be recognized so it is kept out of the
    # extraction input; feeding it derails convention extraction on small models.
    assert _is_mechanical_lesson("Success: domain=file; action=create; files=adder.py")
    assert _is_mechanical_lesson("Task failed: timeout. Consider alternative approaches.")
    assert _is_mechanical_lesson("  Success: domain=code; action=edit")  # leading space


def test_natural_lessons_kept():
    # A genuine, human-meaningful lesson is not telemetry and stays in the input.
    assert not _is_mechanical_lesson("The build only passes when run from the repo root.")
    assert not _is_mechanical_lesson("Use the async client; the sync one deadlocks.")


def test_extraction_token_budget_has_reasoning_headroom():
    # Reasoning fast-tier models (e.g. gpt-5-mini) spend tokens on reasoning
    # before any content. At 512 they return empty and extraction yields nothing,
    # silently disabling all consolidation. Keep enough headroom for both.
    assert _EXTRACTION_MAX_TOKENS >= 2048


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

    # Fake episode + store + manager. utility>0 is a successful run, so durable
    # facts are allowed (failures are gated out; see the success-gate test below).
    class _Ep:
        id = "ep1"
        task_summary = "add a geometry helper"
        result = "done"
        lessons = ""
        entities = ""
        utility = 1

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


@pytest.mark.asyncio
async def test_failed_episode_writes_no_durable_facts(monkeypatch):
    """A FAILED run (utility<0) must not write decisions/lessons/conventions:
    its result text is the model's own (often false) self-assessment, so
    extracting facts from it poisons memory with false 'success' knowledge."""
    import rune.memory.consolidation as consol

    class _Ep:
        id = "epf"
        task_summary = "implement apply_discount"
        # The model's FALSE self-report on a run whose tests actually failed:
        result = "Implemented per the rules. The tests passed successfully."
        lessons = ""
        entities = ""
        utility = -1  # verified failure

    class _Store:
        conn = type("C", (), {"execute": lambda *a, **k: None})()

        def get_recent_episodes(self, limit=50):
            return [_Ep()]

    class _Mgr:
        store = _Store()

    import rune.memory.manager as mgr_mod
    monkeypatch.setattr(mgr_mod, "get_memory_manager", lambda: _Mgr())

    class _Client:
        async def completion(self, **kw):
            return {"choices": [{"message": {"content": (
                '{"commitments": [], "lessons": ["the tests passed successfully"], '
                '"entities": [], "decisions": ["the discount logic is correct"], '
                '"conventions": ["members get an extra 5% discount"]}'
            )}}]}

    import rune.llm.client as client_mod
    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _Client())

    saved: list[tuple] = []
    import rune.memory.markdown_store as md_store
    monkeypatch.setattr(md_store, "save_learned_fact",
                        lambda cat, key, val, conf: saved.append((cat, key, val, conf)))
    import rune.memory.state as state_mod
    monkeypatch.setattr(state_mod, "is_suppressed", lambda *_a, **_k: False)

    await consol.consolidate_episode("epf")

    assert saved == [], f"failed episode wrote durable facts (poison): {saved}"
