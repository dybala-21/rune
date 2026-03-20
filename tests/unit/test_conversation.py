"""Tests for the conversation module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rune.conversation.manager import ConversationManager
from rune.conversation.store import ConversationStore
from rune.conversation.types import Conversation, ConversationTurn


class TestConversationTurn:
    def test_conversation_turn_defaults(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.timestamp is not None
        assert turn.tool_calls == []

    def test_turn_with_tool_calls(self):
        turn = ConversationTurn(
            role="assistant",
            content="I'll read that file",
            tool_calls=[{"name": "file_read", "args": {"path": "/tmp/a.py"}}],
        )
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0]["name"] == "file_read"


class TestConversation:
    def test_auto_generated_id(self):
        conv = Conversation(user_id="u1")
        assert len(conv.id) == 16
        assert conv.user_id == "u1"
        assert conv.title == ""
        assert conv.turns == []
        assert conv.digest == ""

    def test_unique_ids(self):
        c1 = Conversation(user_id="u1")
        c2 = Conversation(user_id="u1")
        assert c1.id != c2.id


class TestConversationManager:
    def test_conversation_manager_start(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")
        assert isinstance(conv, Conversation)
        assert conv.id  # auto-generated
        assert conv.user_id == "u1"
        # Title starts empty, set on first user turn
        assert conv.title == ""

    def test_add_turn_sets_title(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")

        turn = mgr.add_turn(conv.id, "user", "Fix the login bug")
        assert turn.role == "user"
        assert conv.title == "Fix the login bug"

    def test_add_turn_title_not_overwritten(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")

        mgr.add_turn(conv.id, "user", "First message")
        mgr.add_turn(conv.id, "user", "Second message")
        assert conv.title == "First message"

    def test_add_turn_assistant_does_not_set_title(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")

        mgr.add_turn(conv.id, "assistant", "Hello, how can I help?")
        assert conv.title == ""

    def test_add_turn_unknown_conversation_raises(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)

        with pytest.raises(KeyError, match="not active"):
            mgr.add_turn("nonexistent-id", "user", "hello")

    def test_context_window_respects_budget(self):
        store = MagicMock()
        mgr = ConversationManager(store=store, token_budget=50)
        conv = mgr.start_conversation(user_id="u1")

        # Add many turns that will exceed the budget
        for i in range(20):
            mgr.add_turn(conv.id, "user", f"Message {i} " * 10)

        window = mgr.get_context_window(conv.id)
        # Window should be smaller than total turns
        assert len(window) < 20

    def test_context_window_keeps_system_turns(self):
        store = MagicMock()
        mgr = ConversationManager(store=store, token_budget=500)
        conv = mgr.start_conversation(user_id="u1")

        # Manually add a system turn
        sys_turn = ConversationTurn(role="system", content="You are an assistant")
        conv.turns.append(sys_turn)
        mgr.add_turn(conv.id, "user", "Hello")

        window = mgr.get_context_window(conv.id)
        roles = [t.role for t in window]
        assert "system" in roles

    def test_context_window_empty_conversation(self):
        store = MagicMock()
        mgr = ConversationManager(store=store)
        conv = mgr.start_conversation(user_id="u1")
        assert mgr.get_context_window(conv.id) == []


class TestConversationStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)

            conv = Conversation(user_id="u1", title="Test Conv")
            conv.turns.append(ConversationTurn(role="user", content="hello"))
            conv.turns.append(ConversationTurn(role="assistant", content="hi there"))

            await store.save(conv)
            loaded = await store.load(conv.id)

            assert loaded is not None
            assert loaded.user_id == "u1"
            assert loaded.title == "Test Conv"
            assert len(loaded.turns) == 2
            assert loaded.turns[0].content == "hello"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)
            assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)

            c1 = Conversation(user_id="u1", title="First")
            c2 = Conversation(user_id="u1", title="Second")
            c3 = Conversation(user_id="u2", title="Other user")

            await store.save(c1)
            await store.save(c2)
            await store.save(c3)

            u1_convs = await store.list("u1")
            assert len(u1_convs) == 2

            u2_convs = await store.list("u2")
            assert len(u2_convs) == 1

    @pytest.mark.asyncio
    async def test_delete_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)

            conv = Conversation(user_id="u1", title="Delete me")
            conv.turns.append(ConversationTurn(role="user", content="bye"))
            await store.save(conv)

            await store.delete(conv.id)
            assert await store.load(conv.id) is None

    def test_generate_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)

            conv = Conversation(user_id="u1")
            conv.turns.append(ConversationTurn(role="user", content="Fix the bug"))
            conv.turns.append(ConversationTurn(
                role="assistant", content="Done",
                tool_calls=[{"name": "file_edit"}],
            ))

            digest = store._generate_digest(conv)
            assert "Fix the bug" in digest
            assert "2 turns" in digest
            assert "1 tool calls" in digest

    def test_generate_digest_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)
            conv = Conversation(user_id="u1")
            assert store._generate_digest(conv) == ""
