"""Regression tests for conversation-history quality defects.

(a) store-time truncation was too lossy at 2000 tokens (raised to 8000),
(w) one oversized turn in the window builder dropped ALL older context,
(c) an empty assistant answer left a dangling user turn in history,
(d) compaction raced concurrent add_turn during the summarize await and
    silently dropped the turns appended in that window.
"""

from __future__ import annotations

import asyncio

import pytest

from rune.conversation.manager import (
    MAX_ASSISTANT_TOKENS,
    ConversationManager,
)
from rune.conversation.store import ConversationStore
from rune.utils.tokenizer import count_tokens


@pytest.fixture
def manager(tmp_path):
    return ConversationManager(ConversationStore(tmp_path / "conv.db"))


def _long_text(tokens: int) -> str:
    # ~1 token per word for simple ASCII words
    return " ".join(f"word{i}" for i in range(tokens))


# (a) store-time truncation


def test_assistant_turns_under_8k_stored_intact(manager):
    conv = manager.start_conversation("t")
    content = _long_text(1200)  # ~3000 tokens: over the old 2000 cap, under 8000
    manager.add_turn(conv.id, "assistant", content)
    stored = conv.turns[-1].content
    assert "truncated" not in stored
    assert stored == content


def test_assistant_turns_over_cap_still_bounded(manager):
    conv = manager.start_conversation("t")
    manager.add_turn(conv.id, "assistant", _long_text(MAX_ASSISTANT_TOKENS * 2))
    stored = conv.turns[-1].content
    assert "truncated" in stored
    assert count_tokens(stored) <= MAX_ASSISTANT_TOKENS + 100


# (w) oversized turn in the window builder


def test_oversized_middle_turn_does_not_evict_older_context(manager):
    conv = manager.start_conversation("t")
    manager.add_turn(conv.id, "user", "first question about herons")
    manager.add_turn(conv.id, "assistant", "herons are birds")
    # Oversized user turn (user turns are not store-truncated)
    manager.add_turn(conv.id, "user", _long_text(6000))
    manager.add_turn(conv.id, "assistant", "short reply")
    manager.add_turn(conv.id, "user", "and a follow-up")

    msgs = manager._build_messages_within_budget(conv.id, budget_tokens=2000)
    joined = str(msgs)
    # Old behavior: the 6000-token turn broke the loop and dropped the heron
    # turns entirely. New behavior: placeholder + older turns kept.
    assert "herons" in joined
    assert "omitted" in joined


def test_oversized_latest_turn_included_truncated(manager):
    conv = manager.start_conversation("t")
    manager.add_turn(conv.id, "user", _long_text(6000) + " FINAL_MARKER")

    msgs = manager._build_messages_within_budget(conv.id, budget_tokens=1500)
    assert msgs, "latest turn must not vanish when it exceeds the budget"
    assert "FINAL_MARKER" in msgs[-1]["content"]
    assert count_tokens(msgs[-1]["content"]) <= 1500


# (c) empty assistant answer


async def test_empty_answer_records_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    from rune.api import conversation_wiring

    conversation_wiring._reset_for_tests()
    manager = conversation_wiring.get_conv_manager()
    conv_id = await conversation_wiring.resolve_conversation(
        manager, "web_empty", sticky=False,
    )
    conversation_wiring.record_user_turn(manager, conv_id, "do the thing")

    class _Loop:
        _last_answer_text = ""
        _last_goal_type = "chat"

    await conversation_wiring.record_assistant_turn(
        manager, conv_id, _Loop(), "", reason="max_steps",
    )
    turns = manager._active[conv_id].turns
    assert [t.role for t in turns] == ["user", "assistant"]
    assert "max_steps" in turns[-1].content
    conversation_wiring._reset_for_tests()


# (d) compaction race


async def test_compaction_keeps_turns_added_during_summarize(manager, monkeypatch):
    conv = manager.start_conversation("t")
    for i in range(10):
        manager.add_turn(conv.id, "user", f"question {i} " + _long_text(300))
        manager.add_turn(conv.id, "assistant", f"answer {i} " + _long_text(300))

    async def _slow_summarize(self, turns):
        # A turn arrives mid-compaction (the old code recomputed the preserve
        # slice by tail count and silently dropped turns in this window).
        manager.add_turn(conv.id, "user", "RACE_TURN arrived during compaction")
        await asyncio.sleep(0)
        return "summary of earlier turns"

    monkeypatch.setattr(ConversationManager, "summarize_turns", _slow_summarize)

    await manager.compact_conversation(conv.id, budget_tokens=1000)

    contents = [t.content for t in conv.turns]
    assert any("RACE_TURN" in c for c in contents), (
        "turn added during compaction was dropped"
    )
    assert conv.turns[0].role == "system"  # summary took the old prefix


async def test_concurrent_compactions_serialized(manager, monkeypatch):
    conv = manager.start_conversation("t")
    # ~100 tokens per turn: 12 turns (~1200) exceed the 65% threshold of a
    # 1000-token budget; after one compaction (summary + 5 turns ≈ 520) the
    # second run is below threshold and must skip.
    for _i in range(12):
        manager.add_turn(conv.id, "user", _long_text(40))

    calls = []

    async def _summarize(self, turns):
        calls.append(len(turns))
        await asyncio.sleep(0.01)
        return "summary"

    monkeypatch.setattr(ConversationManager, "summarize_turns", _summarize)
    await asyncio.gather(
        manager.compact_conversation(conv.id, budget_tokens=1000),
        manager.compact_conversation(conv.id, budget_tokens=1000),
    )
    # Second compaction sees the already-compacted (small) conversation and
    # skips; without the lock both would summarize the same prefix.
    assert len(calls) == 1
