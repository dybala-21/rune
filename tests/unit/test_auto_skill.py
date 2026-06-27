"""Tests for auto-skill reuse wiring (default-off flag).

The auto-skill system was built but not wired in. It is gated on verification:
distil a skill only from a completed/verified run, and inject a matching one into
a later task. These tests cover the wiring; the flag is off by default, so they
set it explicitly.
"""

from __future__ import annotations

import pytest

from rune.agent.loop import NativeAgentLoop, _auto_skill_enabled
from rune.agent.memory_bridge import ToolTraceEntry
from rune.agent.prompts import build_system_prompt
from rune.config.schema import RuneConfig, SkillsConfig
from rune.skills.registry import get_skill_registry
from rune.types import CompletionTrace

# Config + flag


def test_skills_config_default_off_with_alias():
    assert SkillsConfig().auto_skill is False
    assert SkillsConfig(**{"autoSkill": True}).auto_skill is True
    assert RuneConfig().skills.auto_skill is False  # off by default at the root


def test_auto_skill_enabled_env_overrides_config(monkeypatch):
    from rune.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.skills, "auto_skill", False)

    monkeypatch.setenv("RUNE_AUTO_SKILL", "1")
    assert _auto_skill_enabled() is True
    monkeypatch.setenv("RUNE_AUTO_SKILL", "off")
    assert _auto_skill_enabled() is False  # env wins over config

    monkeypatch.delenv("RUNE_AUTO_SKILL", raising=False)
    assert _auto_skill_enabled() is False  # config default
    monkeypatch.setattr(cfg.skills, "auto_skill", True)
    assert _auto_skill_enabled() is True


# Prompt rendering


def test_build_system_prompt_renders_skill_section():
    prompt = build_system_prompt(goal="x", skill_context="STEP 1: read the file")
    assert "## Learned Skill" in prompt
    assert "STEP 1: read the file" in prompt


def test_build_system_prompt_omits_skill_section_when_absent():
    assert "## Learned Skill" not in build_system_prompt(goal="x")


# Loop wiring (no LLM — pattern extraction is local; refiner=None)


def _loop(auto_skill: bool) -> NativeAgentLoop:
    loop = NativeAgentLoop()
    loop._auto_skill = auto_skill
    return loop


def test_build_skill_context_off_returns_none():
    assert _loop(False)._build_skill_context("read a file", "code") is None


def test_build_skill_context_only_for_code_or_full():
    loop = _loop(True)
    # chat-category goals never get a skill injection even with the flag on
    assert loop._build_skill_context("hello there", "chat") is None


@pytest.mark.asyncio
async def test_distill_then_inject_roundtrip(monkeypatch):
    """A verified run distils a skill; a later matching goal retrieves it."""
    # Isolate the registry singleton for this test.
    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})

    loop = _loop(True)
    loop._tool_trace = [
        ToolTraceEntry(
            tool_name="file_read", params={"file_path": "/proj/parser.py"}, success=True
        ),
        ToolTraceEntry(
            tool_name="file_edit", params={"file_path": "/proj/parser.py"}, success=True
        ),
    ]
    # reason="verified" + high evidence + few steps => quality gate passes
    trace = CompletionTrace(
        reason="verified",
        final_step=2,
        total_tokens_used=3000,
        evidence_score=0.9,
    )

    await loop._maybe_distill_skill("Fix the bug in the parser module", trace)

    # A skill was registered from the verified run...
    assert len(reg._skills) == 1
    registered = next(iter(reg._skills.values()))
    assert registered.author == "auto"

    # ...and a later similar goal retrieves it for injection.
    injected = loop._build_skill_context("fix the parser module bug", "code")
    assert injected is not None and len(injected) > 0


@pytest.mark.asyncio
async def test_failed_run_distills_nothing(monkeypatch):
    """The verification quality gate blocks skill capture from a failed run."""
    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})
    loop = _loop(True)
    loop._tool_trace = [
        ToolTraceEntry(tool_name="file_edit", params={"file_path": "/a.py"}, success=False),
    ]
    # A non-completed reason => quality 0.0 => no skill (unlike "what worked").
    await loop._maybe_distill_skill("do the thing", CompletionTrace(reason="stalled"))
    assert len(reg._skills) == 0


# --- Strategic skill body synthesis (distilled skills are reusable procedures,
# not verbatim tool-call transcripts that an agent gains nothing from) ---


class _FakeRefiner:
    def __init__(self, text):
        self._text = text

    async def refine(self, prompt, max_tokens=600):
        return self._text


@pytest.mark.asyncio
async def test_synthesize_strategic_body_none_without_refiner():
    from rune.agent.memory_bridge import _synthesize_strategic_body

    steps = [{"tool": "file_write", "params_template": {}}]
    assert await _synthesize_strategic_body("g", steps, "generate", None) is None


@pytest.mark.asyncio
async def test_synthesize_strategic_body_rejects_non_procedure():
    from rune.agent.memory_bridge import _synthesize_strategic_body

    steps = [{"tool": "file_write", "params_template": {}}]
    # An echo / empty-ish reply lacking the asked-for "## Procedure" is rejected.
    out = await _synthesize_strategic_body("g", steps, "generate", _FakeRefiner("sure, ok"))
    assert out is None


@pytest.mark.asyncio
async def test_synthesize_strategic_body_accepts_procedure():
    from rune.agent.memory_bridge import _synthesize_strategic_body

    steps = [{"tool": "file_write", "params_template": {}}]
    good = "## When to use\nadding a function to a module\n## Procedure\n1. file_write <target>\n## Verify\nrun tests"
    out = await _synthesize_strategic_body("g", steps, "generate", _FakeRefiner(good))
    assert out is not None and "## Procedure" in out


@pytest.mark.asyncio
async def test_distilled_body_is_strategic_when_refiner_present(monkeypatch):
    from types import SimpleNamespace

    from rune.agent.memory_bridge import maybe_generate_skill
    from rune.skills.registry import get_skill_registry

    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})
    trace = [
        ToolTraceEntry(
            tool_name="file_write",
            params={"path": "x.py", "content": "def f(): pass"},
            success=True,
        ),
        ToolTraceEntry(tool_name="bash", params={"command": "pytest"}, success=True),
    ]
    res = SimpleNamespace(reason="completed", evidence_score=0.9, steps=2)
    good = "## When to use\nwriting a module + test\n## Procedure\n1. file_write <target_file>: create the module\n## Verify\nrun pytest"
    sk = await maybe_generate_skill(
        goal="make x.py and test it", result=res, trace=trace, refiner=_FakeRefiner(good)
    )
    body = get_skill_registry().get(sk["name"]).body
    assert "## Procedure" in body and "<target_file>" in body
    assert "## Step 1:" not in body  # not the verbatim transcript


@pytest.mark.asyncio
async def test_distilled_body_falls_back_when_refiner_fails(monkeypatch):
    from types import SimpleNamespace

    from rune.agent.memory_bridge import maybe_generate_skill
    from rune.skills.registry import get_skill_registry

    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})
    trace = [
        ToolTraceEntry(tool_name="file_write", params={"path": "x.py"}, success=True),
        ToolTraceEntry(tool_name="bash", params={"command": "pytest"}, success=True),
    ]
    res = SimpleNamespace(reason="completed", evidence_score=0.9, steps=2)
    sk = await maybe_generate_skill(
        goal="make x.py and run its tests",
        result=res,
        trace=trace,
        refiner=_FakeRefiner("junk no procedure"),
    )
    body = get_skill_registry().get(sk["name"]).body
    assert "## Step 1:" in body  # deterministic fallback preserved


@pytest.mark.asyncio
async def test_auto_skill_persists_distilled_skill_for_cross_session_reuse(monkeypatch):
    """auto_skill (not just gated_learning) must persist the skill to disk, else
    it dies with the one-shot process and is never reused next session — the
    documented cause of 'no cross-session lift'."""
    from types import SimpleNamespace

    from rune.agent.memory_bridge import maybe_generate_skill
    from rune.config import get_config
    from rune.skills.registry import get_skill_registry

    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})
    monkeypatch.setattr(get_config().skills, "auto_skill", True, raising=False)
    monkeypatch.setattr(get_config().skills, "gated_learning", False, raising=False)
    written = []
    import rune.skills.persistence as persist

    monkeypatch.setattr(persist, "write_skill_to_disk", lambda s: written.append(s.name) or "/x")

    trace = [
        ToolTraceEntry(tool_name="file_write", params={"path": "x.py"}, success=True),
        ToolTraceEntry(tool_name="bash", params={"command": "pytest"}, success=True),
    ]
    await maybe_generate_skill(
        goal="make x.py and run its tests",
        result=SimpleNamespace(reason="completed", evidence_score=0.9, steps=2),
        trace=trace,
    )
    assert written, "distilled skill was not persisted under auto_skill"


@pytest.mark.asyncio
async def test_no_persist_when_both_off(monkeypatch):
    from types import SimpleNamespace

    from rune.agent.memory_bridge import maybe_generate_skill
    from rune.config import get_config
    from rune.skills.registry import get_skill_registry

    reg = get_skill_registry()
    monkeypatch.setattr(reg, "_skills", {})
    monkeypatch.setattr(get_config().skills, "auto_skill", False, raising=False)
    monkeypatch.setattr(get_config().skills, "gated_learning", False, raising=False)
    written = []
    import rune.skills.persistence as persist

    monkeypatch.setattr(persist, "write_skill_to_disk", lambda s: written.append(s.name) or "/x")
    trace = [
        ToolTraceEntry(tool_name="file_write", params={"path": "x.py"}, success=True),
        ToolTraceEntry(tool_name="bash", params={"command": "pytest"}, success=True),
    ]
    await maybe_generate_skill(
        goal="make x.py and run its tests",
        result=SimpleNamespace(reason="completed", evidence_score=0.9, steps=2),
        trace=trace,
    )
    assert not written  # behaviour-neutral when learning is off
