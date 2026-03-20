"""Tests for new ConversationStore methods."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rune.conversation.store import ConversationStore
from rune.conversation.types import Conversation, ConversationTurn


@pytest.fixture
def store():
    """Create a temporary ConversationStore for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conv.db"
        yield ConversationStore(db_path)


def _make_conversation(
    user_id: str = "u1",
    title: str = "Test",
    status: str = "active",
    execution_context: str = "",
    num_turns: int = 0,
    channel: str = "",
) -> Conversation:
    """Helper to build a conversation with optional turns."""
    conv = Conversation(
        user_id=user_id,
        title=title,
        status=status,
        execution_context=execution_context,
    )
    for i in range(num_turns):
        role = "user" if i % 2 == 0 else "assistant"
        conv.turns.append(ConversationTurn(
            role=role,
            content=f"Turn {i}",
            channel=channel,
        ))
    return conv


class TestGetTurnCount:
    @pytest.mark.asyncio
    async def test_zero_turns(self, store):
        conv = _make_conversation()
        await store.save(conv)
        count = await store.get_turn_count(conv.id)
        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_turns(self, store):
        conv = _make_conversation(num_turns=5)
        await store.save(conv)
        count = await store.get_turn_count(conv.id)
        assert count == 5

    @pytest.mark.asyncio
    async def test_nonexistent_conversation(self, store):
        count = await store.get_turn_count("nonexistent")
        assert count == 0


class TestGetLastTurn:
    @pytest.mark.asyncio
    async def test_returns_last_turn(self, store):
        conv = _make_conversation(num_turns=3)
        await store.save(conv)
        last = await store.get_last_turn(conv.id)
        assert last is not None
        assert last.content == "Turn 2"
        assert last.role == "user"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_turns(self, store):
        conv = _make_conversation()
        await store.save(conv)
        last = await store.get_last_turn(conv.id)
        assert last is None

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent(self, store):
        last = await store.get_last_turn("nonexistent")
        assert last is None

    @pytest.mark.asyncio
    async def test_last_turn_fields(self, store):
        conv = _make_conversation()
        turn = ConversationTurn(
            role="assistant",
            content="reply",
            channel="slack",
            episode_id="ep1",
            execution_context="ctx",
        )
        conv.turns.append(turn)
        await store.save(conv)

        last = await store.get_last_turn(conv.id)
        assert last is not None
        assert last.role == "assistant"
        assert last.content == "reply"
        assert last.channel == "slack"
        assert last.episode_id == "ep1"
        assert last.execution_context == "ctx"


class TestReplaceTurnsWithSummary:
    @pytest.mark.asyncio
    async def test_replaces_older_turns(self, store):
        conv = _make_conversation(num_turns=10)
        await store.save(conv)

        replaced = await store.replace_turns_with_summary(conv.id, "Summary of earlier turns", keep_latest=3)
        assert replaced == 7

        # Should now have 3 kept + 1 summary = 4 turns
        count = await store.get_turn_count(conv.id)
        assert count == 4

    @pytest.mark.asyncio
    async def test_summary_turn_content(self, store):
        conv = _make_conversation(num_turns=8)
        await store.save(conv)

        await store.replace_turns_with_summary(conv.id, "Test summary text", keep_latest=3)

        # Reload and check the summary turn exists
        loaded = await store.load(conv.id)
        assert loaded is not None
        summary_turns = [t for t in loaded.turns if "Summary of" in t.content]
        assert len(summary_turns) == 1
        assert "Test summary text" in summary_turns[0].content
        assert summary_turns[0].role == "system"

    @pytest.mark.asyncio
    async def test_no_replacement_when_few_turns(self, store):
        conv = _make_conversation(num_turns=3)
        await store.save(conv)

        replaced = await store.replace_turns_with_summary(conv.id, "Summary", keep_latest=5)
        assert replaced == 0

        count = await store.get_turn_count(conv.id)
        assert count == 3

    @pytest.mark.asyncio
    async def test_no_replacement_when_equal_to_keep(self, store):
        conv = _make_conversation(num_turns=5)
        await store.save(conv)

        replaced = await store.replace_turns_with_summary(conv.id, "Summary", keep_latest=5)
        assert replaced == 0


class TestArchiveStale:
    @pytest.mark.asyncio
    async def test_archives_old_conversations(self, store):
        # Create a conversation with old updated_at
        conv = _make_conversation()
        conv.updated_at = datetime.now() - timedelta(hours=2)
        await store.save(conv)

        archived = await store.archive_stale(idle_minutes=30)
        assert archived >= 1

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.status == "archived"

    @pytest.mark.asyncio
    async def test_does_not_archive_recent(self, store):
        conv = _make_conversation()
        # updated_at is set to now by default
        await store.save(conv)

        archived = await store.archive_stale(idle_minutes=30)
        assert archived == 0

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.status == "active"

    @pytest.mark.asyncio
    async def test_archive_stale_with_user_filter(self, store):
        # Two conversations from different users, both old
        c1 = _make_conversation(user_id="u1")
        c1.updated_at = datetime.now() - timedelta(hours=2)
        c2 = _make_conversation(user_id="u2")
        c2.updated_at = datetime.now() - timedelta(hours=2)

        await store.save(c1)
        await store.save(c2)

        archived = await store.archive_stale(idle_minutes=30, user_id="u1")
        assert archived >= 1

        # u1's conversation should be archived
        loaded_c1 = await store.load(c1.id)
        assert loaded_c1 is not None
        assert loaded_c1.status == "archived"

        # u2's conversation should still be active
        loaded_c2 = await store.load(c2.id)
        assert loaded_c2 is not None
        assert loaded_c2.status == "active"


class TestArchiveConversation:
    @pytest.mark.asyncio
    async def test_archives_single_conversation(self, store):
        conv = _make_conversation()
        await store.save(conv)

        await store.archive_conversation(conv.id)

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.status == "archived"

    @pytest.mark.asyncio
    async def test_archive_nonexistent_does_not_error(self, store):
        # Should not raise
        await store.archive_conversation("nonexistent")


class TestFindActiveConversation:
    @pytest.mark.asyncio
    async def test_finds_active_conversation(self, store):
        conv = _make_conversation(user_id="u1", title="Active one")
        await store.save(conv)

        found = await store.find_active_conversation("u1")
        assert found is not None
        assert found.id == conv.id
        assert found.title == "Active one"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active(self, store):
        conv = _make_conversation(user_id="u1", status="archived")
        await store.save(conv)

        found = await store.find_active_conversation("u1")
        assert found is None

    @pytest.mark.asyncio
    async def test_returns_most_recent_active(self, store):
        c1 = _make_conversation(user_id="u1", title="Older")
        c1.updated_at = datetime.now() - timedelta(hours=1)
        await store.save(c1)

        c2 = _make_conversation(user_id="u1", title="Newer")
        await store.save(c2)

        found = await store.find_active_conversation("u1")
        assert found is not None
        assert found.id == c2.id

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_user(self, store):
        found = await store.find_active_conversation("unknown_user")
        assert found is None


class TestUpdateConversation:
    @pytest.mark.asyncio
    async def test_update_title(self, store):
        conv = _make_conversation(title="Old title")
        await store.save(conv)

        await store.update_conversation(conv.id, title="New title")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.title == "New title"

    @pytest.mark.asyncio
    async def test_update_status(self, store):
        conv = _make_conversation()
        await store.save(conv)

        await store.update_conversation(conv.id, status="archived")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.status == "archived"

    @pytest.mark.asyncio
    async def test_update_execution_context(self, store):
        conv = _make_conversation()
        await store.save(conv)

        await store.update_conversation(conv.id, execution_context="ctx:test")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.execution_context == "ctx:test"

    @pytest.mark.asyncio
    async def test_update_digest(self, store):
        conv = _make_conversation()
        await store.save(conv)

        await store.update_conversation(conv.id, digest="new digest")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.digest == "new digest"

    @pytest.mark.asyncio
    async def test_ignores_disallowed_fields(self, store):
        conv = _make_conversation(title="Original")
        await store.save(conv)

        # 'user_id' is not in allowed fields
        await store.update_conversation(conv.id, user_id="hacker", title="Updated")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.user_id == "u1"
        assert loaded.title == "Updated"

    @pytest.mark.asyncio
    async def test_update_sets_updated_at(self, store):
        conv = _make_conversation()
        conv.updated_at = datetime.now() - timedelta(hours=5)
        await store.save(conv)

        old_updated = conv.updated_at
        await store.update_conversation(conv.id, title="Touched")

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.updated_at > old_updated


class TestNewColumnFields:
    @pytest.mark.asyncio
    async def test_status_field_persists(self, store):
        conv = _make_conversation(status="archived")
        await store.save(conv)

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.status == "archived"

    @pytest.mark.asyncio
    async def test_execution_context_field_persists(self, store):
        conv = _make_conversation(execution_context="daemon:main")
        await store.save(conv)

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.execution_context == "daemon:main"

    @pytest.mark.asyncio
    async def test_turn_channel_persists(self, store):
        conv = _make_conversation()
        conv.turns.append(ConversationTurn(role="user", content="hi", channel="slack"))
        await store.save(conv)

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.turns[0].channel == "slack"

    @pytest.mark.asyncio
    async def test_turn_archived_flag_persists(self, store):
        conv = _make_conversation()
        conv.turns.append(ConversationTurn(role="user", content="old", archived=True))
        conv.turns.append(ConversationTurn(role="user", content="new", archived=False))
        await store.save(conv)

        loaded = await store.load(conv.id)
        assert loaded is not None
        assert loaded.turns[0].archived is True
        assert loaded.turns[1].archived is False

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, store):
        c1 = _make_conversation(user_id="u1", title="Active", status="active")
        c2 = _make_conversation(user_id="u1", title="Archived", status="archived")
        await store.save(c1)
        await store.save(c2)

        active = await store.list("u1", status="active")
        assert len(active) == 1
        assert active[0].title == "Active"

        archived = await store.list("u1", status="archived")
        assert len(archived) == 1
        assert archived[0].title == "Archived"
