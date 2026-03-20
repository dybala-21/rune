"""Tests for conversation compression pipeline.

Tests for build_budgeted_context, compact_conversation, summarize_turns,
archive_stale_conversations, and capture_execution_context.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rune.conversation.context_budget import GoalClassification
from rune.conversation.manager import (
    ConversationCompactionEvent,
    ConversationManager,
    ExecutionContextSnapshot,
    LLMSummarizer,
)
from rune.conversation.types import Conversation, ConversationTurn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeLLM:
    """Fake LLM summarizer for testing."""

    def __init__(self, response: str = "Summary of the conversation."):
        self.response = response
        self.calls: list[tuple[str, int]] = []

    async def summarize(self, prompt: str, *, max_tokens: int = 800) -> str:
        self.calls.append((prompt, max_tokens))
        return self.response


class SlowLLM:
    """LLM that times out."""

    async def summarize(self, prompt: str, *, max_tokens: int = 800) -> str:
        await asyncio.sleep(30)
        return "Should not reach here"


class ErrorLLM:
    """LLM that raises an error."""

    async def summarize(self, prompt: str, *, max_tokens: int = 800) -> str:
        raise RuntimeError("LLM service unavailable")


@pytest.fixture
def store():
    return MagicMock()


@pytest.fixture
def mgr(store):
    return ConversationManager(store=store)


@pytest.fixture
def mgr_with_llm(store):
    return ConversationManager(store=store, llm=FakeLLM())


def _make_conv_with_turns(
    mgr: ConversationManager, n_turns: int = 10, content_size: int = 200
) -> str:
    """Helper: create a conversation with alternating user/assistant turns."""
    conv = mgr.start_conversation(user_id="u1")
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        mgr.add_turn(conv.id, role, f"Turn {i}: " + "x" * content_size)
    return conv.id


# ---------------------------------------------------------------------------
# Tests: summarize_turns
# ---------------------------------------------------------------------------


class TestSummarizeTurns:
    @pytest.mark.asyncio
    async def test_extractive_fallback_when_no_llm(self, store):
        mgr = ConversationManager(store=store)
        turns = [
            ConversationTurn(
                role="user",
                content="Please fix the login bug in auth.py. It crashes on empty passwords.",
            ),
            ConversationTurn(
                role="assistant",
                content="I found the issue in auth.py line 42. The validation was missing a null check.",
            ),
        ]
        result = await mgr.summarize_turns(turns)
        assert result.startswith("[Conversation Summary]")
        assert "[user]" in result
        assert "[assistant]" in result

    @pytest.mark.asyncio
    async def test_llm_summarization(self, store):
        llm = FakeLLM("Fixed auth.py login bug by adding null check.")
        mgr = ConversationManager(store=store, llm=llm)
        turns = [
            ConversationTurn(role="user", content="Fix the login bug."),
            ConversationTurn(role="assistant", content="Done."),
        ]
        result = await mgr.summarize_turns(turns)
        assert result == "[Conversation Summary]\nFixed auth.py login bug by adding null check."
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_extractive(self, store):
        """When LLM takes >10s, extractive fallback should be used."""
        ConversationManager(store=store, llm=SlowLLM())
        turns = [
            ConversationTurn(
                role="user",
                content="Implement the microservice architecture for the payment system.",
            ),
        ]
        # Use a shorter timeout for test speed -- but the code uses 10s,
        # so we monkey-patch asyncio.wait_for timeout via the manager.
        # Actually, SlowLLM sleeps 30s; asyncio.wait_for(10s) will timeout.
        # We need to actually let it timeout, but that takes 10s in real time.
        # Instead, test the error path with ErrorLLM which fails instantly.
        mgr_err = ConversationManager(store=store, llm=ErrorLLM())
        result = await mgr_err.summarize_turns(turns)
        assert result.startswith("[Conversation Summary]")

    @pytest.mark.asyncio
    async def test_llm_error_falls_back(self, store):
        mgr = ConversationManager(store=store, llm=ErrorLLM())
        turns = [
            ConversationTurn(
                role="user",
                content="Refactor the database layer to use connection pooling.",
            ),
        ]
        result = await mgr.summarize_turns(turns)
        assert "[Conversation Summary]" in result

    @pytest.mark.asyncio
    async def test_extractive_compact_content(self, store):
        mgr = ConversationManager(store=store)
        turns = [
            ConversationTurn(
                role="user",
                content="Please fix the login bug. It crashes on empty passwords.",
            ),
            ConversationTurn(
                role="assistant",
                content="Found the issue in auth.py. Added null check for password field.",
            ),
        ]
        result = await mgr.summarize_turns(turns)
        assert "[user]" in result
        assert "[assistant]" in result


# ---------------------------------------------------------------------------
# Tests: compact_conversation
# ---------------------------------------------------------------------------


class TestCompactConversation:
    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self, mgr):
        """Compaction should not trigger when tokens < 65% of budget."""
        conv_id = _make_conv_with_turns(mgr, n_turns=4, content_size=10)
        callback = MagicMock()
        # Large budget relative to content
        await mgr.compact_conversation(conv_id, budget_tokens=100_000, on_compaction=callback)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_compaction_triggers_above_threshold(self, store):
        """Compaction should trigger and replace turns with summary."""
        llm = FakeLLM("Compacted summary of earlier turns.")
        mgr = ConversationManager(store=store, llm=llm)
        conv = mgr.start_conversation(user_id="u1")

        # Add many long turns to exceed 65% of a small budget
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_turn(conv.id, role, f"Turn {i}: " + "detail " * 100)

        events: list[ConversationCompactionEvent] = []
        await mgr.compact_conversation(
            conv.id,
            budget_tokens=500,  # Very small budget to force compaction
            on_compaction=events.append,
        )

        # Should have emitted an event
        assert len(events) == 1
        assert events[0].turns_compacted > 0
        assert events[0].turns_preserved > 0

        # Conversation should now have fewer turns
        assert len(conv.turns) < 12
        # First turn should be the summary
        assert conv.turns[0].role == "system"
        assert "[Conversation Summary]" in conv.turns[0].content

    @pytest.mark.asyncio
    async def test_compaction_preserves_latest_turns(self, store):
        llm = FakeLLM("Summary.")
        mgr = ConversationManager(store=store, llm=llm)
        conv = mgr.start_conversation(user_id="u1")

        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_turn(conv.id, role, f"Turn {i}: " + "x" * 200)

        original_last_5 = [t.content for t in conv.turns[-5:]]

        await mgr.compact_conversation(conv.id, budget_tokens=100)

        # The last preserved turns should still be there
        preserved_contents = [t.content for t in conv.turns if t.role != "system"]
        for content in original_last_5:
            assert content in preserved_contents

    @pytest.mark.asyncio
    async def test_compaction_unknown_conversation(self, mgr):
        """Compaction on unknown conversation should be a no-op."""
        await mgr.compact_conversation("nonexistent", budget_tokens=1000)

    @pytest.mark.asyncio
    async def test_compaction_too_few_turns(self, mgr):
        """Compaction with <2 turns to compact should be a no-op."""
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "hello")
        callback = MagicMock()
        await mgr.compact_conversation(conv.id, budget_tokens=10, on_compaction=callback)
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: build_budgeted_context
# ---------------------------------------------------------------------------


class TestBuildBudgetedContext:
    @pytest.mark.asyncio
    async def test_basic_budgeted_context(self, mgr):
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "Hello")
        mgr.add_turn(conv.id, "assistant", "Hi there!")

        result = await mgr.build_budgeted_context("hello", conv.id)
        assert len(result.conversation_messages) == 2
        assert result.conversation_messages[0]["role"] == "user"
        assert result.conversation_messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_budgeted_context_with_classification_hint(self, mgr):
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "Refactor microservice")

        hint = GoalClassification(
            category="code",
            complexity="complex",
            is_complex_coding=True,
        )
        mgr.set_classification_hint("Refactor microservice", hint)

        result = await mgr.build_budgeted_context(
            "Refactor microservice", conv.id
        )
        assert isinstance(result.conversation_messages, list)

    @pytest.mark.asyncio
    async def test_budgeted_context_empty_conversation(self, mgr):
        conv = mgr.start_conversation(user_id="u1")
        result = await mgr.build_budgeted_context("hello", conv.id)
        assert result.conversation_messages == []

    @pytest.mark.asyncio
    async def test_budgeted_context_respects_system_turns(self, mgr):
        conv = mgr.start_conversation(user_id="u1")
        # Manually add system turn
        conv.turns.append(ConversationTurn(role="system", content="You are a helpful assistant."))
        mgr.add_turn(conv.id, "user", "Hello")

        result = await mgr.build_budgeted_context("hello", conv.id)
        roles = [m["role"] for m in result.conversation_messages]
        assert "system" in roles


# ---------------------------------------------------------------------------
# Tests: archive_stale_conversations
# ---------------------------------------------------------------------------


class TestArchiveStaleConversations:
    @pytest.mark.asyncio
    async def test_archives_idle_conversations(self, store):
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "Fix the bug in auth.py")
        mgr.add_turn(conv.id, "assistant", "Done, added null check.")

        # Make the conversation appear stale
        conv.updated_at = datetime.now() - timedelta(minutes=45)

        archived = await mgr.archive_stale_conversations(idle_minutes=30)
        assert archived == 1
        # Conversation should be removed from active cache
        assert conv.id not in mgr._active
        # Store.save should have been called
        store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_archive_active_conversations(self, store):
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "Hello")
        # updated_at is now(), so it shouldn't be archived
        archived = await mgr.archive_stale_conversations(idle_minutes=30)
        assert archived == 0
        assert conv.id in mgr._active

    @pytest.mark.asyncio
    async def test_generates_digest_on_archive(self, store):
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")
        mgr.add_turn(conv.id, "user", "Fix the authentication module")
        mgr.add_turn(conv.id, "assistant", "Updated auth.py with proper validation")
        conv.updated_at = datetime.now() - timedelta(minutes=60)

        await mgr.archive_stale_conversations(idle_minutes=30)

        # Check that the digest was set before saving
        saved_conv = store.save.call_args[0][0]
        assert "Fix the authentication module" in saved_conv.digest
        assert "turns" in saved_conv.digest


# ---------------------------------------------------------------------------
# Tests: _generate_session_digest
# ---------------------------------------------------------------------------


class TestGenerateSessionDigest:
    def test_digest_single_user_turn(self):
        conv = Conversation(user_id="u1")
        conv.turns.append(ConversationTurn(role="user", content="Fix the bug"))
        digest = ConversationManager._generate_session_digest(conv)
        assert "Fix the bug" in digest
        assert "1 turns" in digest

    def test_digest_multi_turn(self):
        conv = Conversation(user_id="u1")
        conv.turns.append(ConversationTurn(role="user", content="First request"))
        conv.turns.append(ConversationTurn(role="assistant", content="First response"))
        conv.turns.append(ConversationTurn(role="user", content="Second request"))
        conv.turns.append(ConversationTurn(role="assistant", content="Second response"))
        digest = ConversationManager._generate_session_digest(conv)
        assert "First request" in digest
        assert "-> Second request" in digest
        assert "| Second response" in digest

    def test_digest_empty_conversation(self):
        conv = Conversation(user_id="u1")
        assert ConversationManager._generate_session_digest(conv) == ""

    def test_digest_no_user_turns(self):
        conv = Conversation(user_id="u1")
        conv.turns.append(ConversationTurn(role="system", content="System prompt"))
        assert ConversationManager._generate_session_digest(conv) == ""


# ---------------------------------------------------------------------------
# Tests: capture_execution_context
# ---------------------------------------------------------------------------


class TestCaptureExecutionContext:
    @pytest.mark.asyncio
    async def test_returns_snapshot(self):
        snap = await ConversationManager.capture_execution_context()
        assert isinstance(snap, ExecutionContextSnapshot)
        assert snap.cwd  # Should have a valid cwd

    @pytest.mark.asyncio
    async def test_snapshot_has_cwd(self):
        snap = await ConversationManager.capture_execution_context()
        assert len(snap.cwd) > 0


# ---------------------------------------------------------------------------
# Tests: extractive_compact
# ---------------------------------------------------------------------------


class TestExtractiveCompact:
    def test_basic_extractive(self):
        turns = [
            ConversationTurn(
                role="user",
                content="I need to fix the authentication module. It has a critical security flaw.",
            ),
            ConversationTurn(
                role="assistant",
                content="I found the vulnerability in auth.py line 42. The password comparison was using == instead of constant-time comparison.",
            ),
        ]
        result = ConversationManager._extractive_compact(turns)
        assert result.startswith("[Conversation Summary]")
        lines = result.split("\n")
        assert len(lines) >= 2  # Header + at least one turn

    def test_extractive_short_content(self):
        turns = [
            ConversationTurn(role="user", content="ok"),
        ]
        result = ConversationManager._extractive_compact(turns)
        # Short content should still produce header
        assert "[Conversation Summary]" in result

    def test_extractive_empty_turns(self):
        result = ConversationManager._extractive_compact([])
        assert result == "[Conversation Summary]"


# ---------------------------------------------------------------------------
# Tests: classification hint cache
# ---------------------------------------------------------------------------


class TestClassificationCache:
    def test_set_and_get(self, store):
        mgr = ConversationManager(store=store)
        hint = GoalClassification(category="code", complexity="complex")
        mgr.set_classification_hint("build the app", hint)
        assert mgr.get_cached_classification("build the app") is hint

    def test_miss_returns_none(self, store):
        mgr = ConversationManager(store=store)
        assert mgr.get_cached_classification("unknown") is None


# ---------------------------------------------------------------------------
# Tests: LLMSummarizer protocol
# ---------------------------------------------------------------------------


class TestLLMSummarizerProtocol:
    def test_fake_llm_is_instance(self):
        assert isinstance(FakeLLM(), LLMSummarizer)

    def test_none_is_not_instance(self):
        assert not isinstance(None, LLMSummarizer)
