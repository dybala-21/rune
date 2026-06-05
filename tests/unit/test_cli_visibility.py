"""Tests for self-improving visibility notes (applied / learned rules)."""

from __future__ import annotations

from rune.agent.memory_bridge import (
    format_applied_rules_note as _applied_rules_note,
)
from rune.agent.memory_bridge import (
    format_learned_rules_note,
)


def test_learned_note_summarises_and_truncates():
    assert format_learned_rules_note([]) is None
    note = format_learned_rules_note([f"r{i}" for i in range(5)])
    assert note is not None and "5 new rule" in note and "more" in note


def test_summarises_learned_rules_block():
    mem = (
        "## Project Context\n- a: b\n\n"
        "## Learned Rules\n"
        "- close_brackets: ensure every bracket is closed\n"
        "- match_expected: output must equal expected\n\n"
        "## Other\n- y: z"
    )
    note = _applied_rules_note(mem)
    assert note is not None
    assert "2 learned rule" in note
    assert "close_brackets" in note and "match_expected" in note
    # the unrelated "## Other" section must NOT leak in
    assert "y" not in note.split(":", 1)[-1]


def test_none_when_no_learned_rules():
    assert _applied_rules_note("## Project Context\n- a: b") is None
    assert _applied_rules_note("") is None


def test_truncates_and_counts_extra():
    rules = "\n".join(f"- rule_{i}: v" for i in range(5))
    note = _applied_rules_note(f"## Learned Rules\n{rules}\n")
    assert note is not None
    assert "5 learned rule" in note
    assert "more" in note  # >3 → "(+2 more)"
