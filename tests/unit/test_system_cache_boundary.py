"""Tests for the cross-turn cache boundary (prompt reorder + provider handling).

The turn-stable instructional prefix must come first, the per-turn dynamic
context (goal, memory, datetime) must come last, and the SYSTEM_CACHE_BOUNDARY
marker must NEVER reach a model — it is an Anthropic breakpoint for Anthropic
and stripped for every other provider.
"""

from __future__ import annotations

from types import SimpleNamespace

from rune.agent.litellm_adapter import _apply_anthropic_cache_control
from rune.agent.prompts import SYSTEM_CACHE_BOUNDARY, build_system_prompt


def _classification():
    return SimpleNamespace(
        goal_type="full",
        intent_categories=frozenset(),
        output_expectation=None,
        is_continuation=False,
        is_complex_coding=False,
        is_multi_task=False,
        requires_execution=False,
    )


def _build(mark: bool) -> str:
    return build_system_prompt(
        goal="UNIQUE_GOAL_TOKEN fix the parser",
        classification=_classification(),
        memory_context="UNIQUE_MEMORY_TOKEN user prefers pytest",
        goal_category="full",
        environment={"cwd": "/tmp/x", "home": "/home/x"},
        mark_cache_boundary=mark,
    )


class TestReorder:
    def test_dynamic_sections_come_after_static(self):
        prompt = _build(mark=True)
        core_pos = prompt.index("Core Principles")  # from PROMPT_CORE (static)
        boundary_pos = prompt.index(SYSTEM_CACHE_BOUNDARY)
        goal_pos = prompt.index("UNIQUE_GOAL_TOKEN")
        mem_pos = prompt.index("UNIQUE_MEMORY_TOKEN")
        env_pos = prompt.index("## Environment")
        # Static persona before the boundary; all dynamic context after it.
        assert core_pos < boundary_pos < env_pos
        assert boundary_pos < goal_pos
        assert boundary_pos < mem_pos

    def test_reorder_applies_even_without_marker(self):
        prompt = _build(mark=False)
        assert SYSTEM_CACHE_BOUNDARY not in prompt
        # Prefix stays stable regardless: static persona precedes dynamic tail.
        assert prompt.index("Core Principles") < prompt.index("## Current Task")
        assert prompt.index("## Environment") < prompt.index("## Current Task")

    def test_content_preserved(self):
        prompt = _build(mark=True).replace(SYSTEM_CACHE_BOUNDARY, "")
        for marker in ("Core Principles", "## Environment", "## Current Task",
                       "UNIQUE_GOAL_TOKEN", "UNIQUE_MEMORY_TOKEN"):
            assert marker in prompt


def _wire_messages(model: str, system: str):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]
    return _apply_anthropic_cache_control(model, msgs)


def _flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return ""


class TestAdapterBoundary:
    def test_anthropic_splits_into_two_cached_blocks(self):
        system = f"STATIC PREFIX{SYSTEM_CACHE_BOUNDARY}DYNAMIC TAIL"
        out = _wire_messages("claude-sonnet-4-5-20250929", system)
        blocks = out[0]["content"]
        assert isinstance(blocks, list) and len(blocks) == 2
        assert blocks[0]["text"] == "STATIC PREFIX"
        assert blocks[1]["text"] == "DYNAMIC TAIL"
        # Both blocks carry a cache breakpoint (prefix + full-system).
        assert all(b.get("cache_control") == {"type": "ephemeral"} for b in blocks)

    def test_anthropic_without_boundary_single_block(self):
        out = _wire_messages("claude-sonnet-4-5-20250929", "PLAIN SYSTEM")
        blocks = out[0]["content"]
        assert isinstance(blocks, list) and len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_strips_boundary_to_plain_string(self):
        system = f"STATIC PREFIX{SYSTEM_CACHE_BOUNDARY}DYNAMIC TAIL"
        out = _wire_messages("gpt-5.4", system)
        content = out[0]["content"]
        assert isinstance(content, str)
        assert SYSTEM_CACHE_BOUNDARY not in content
        assert "STATIC PREFIX" in content and "DYNAMIC TAIL" in content

    def test_boundary_never_reaches_model(self):
        system = f"STATIC{SYSTEM_CACHE_BOUNDARY}DYNAMIC"
        for model in ("claude-opus-4-6", "gpt-5.4", "gemini-2.5-pro", "ollama/llama3.2"):
            out = _wire_messages(model, system)
            assert SYSTEM_CACHE_BOUNDARY not in _flatten_text(out[0]["content"]), model

    def test_non_anthropic_no_boundary_unchanged(self):
        out = _wire_messages("gpt-5.4", "PLAIN SYSTEM")
        assert out[0]["content"] == "PLAIN SYSTEM"
