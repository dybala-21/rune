"""Tests for rune.channels.types — channel data types and adapter ABC."""

from __future__ import annotations

import pytest

from rune.channels.types import (
    ChannelAdapter,
    IncomingMessage,
    Notification,
    OutgoingMessage,
    Priority,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestIncomingMessage:
    def test_construction(self):
        msg = IncomingMessage(channel_id="ch-1", sender_id="u-1", text="hello")
        assert msg.channel_id == "ch-1"
        assert msg.sender_id == "u-1"
        assert msg.text == "hello"
        assert msg.attachments == []
        assert msg.reply_to is None

    def test_with_attachments(self):
        msg = IncomingMessage(
            channel_id="ch-1",
            sender_id="u-1",
            text="see image",
            attachments=[{"type": "image", "url": "https://..."}],
        )
        assert len(msg.attachments) == 1


class TestOutgoingMessage:
    def test_construction(self):
        msg = OutgoingMessage(text="response text")
        assert msg.text == "response text"
        assert msg.reply_to is None

    def test_with_reply_to(self):
        msg = OutgoingMessage(text="reply", reply_to="msg-42")
        assert msg.reply_to == "msg-42"


class TestNotification:
    def test_construction(self):
        n = Notification(channel_id="ch-1", sender_id="u-1", text="alert")
        assert n.priority == Priority.NORMAL

    def test_urgent_priority(self):
        n = Notification(channel_id="ch-1", sender_id="u-1", text="fire", priority=Priority.URGENT)
        assert n.priority == Priority.URGENT


class TestPriority:
    def test_values(self):
        assert Priority.LOW == "low"
        assert Priority.NORMAL == "normal"
        assert Priority.HIGH == "high"
        assert Priority.URGENT == "urgent"

    def test_all_values_are_distinct(self):
        values = [p.value for p in Priority]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# ChannelAdapter ABC
# ---------------------------------------------------------------------------


class _DummyAdapter(ChannelAdapter):
    @property
    def name(self) -> str:
        return "dummy"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        pass

    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None:
        pass

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        return ""


class TestChannelAdapter:
    def test_on_message_default_is_none(self):
        adapter = _DummyAdapter()
        assert adapter.on_message is None

    def test_on_message_setter(self):
        adapter = _DummyAdapter()

        async def handler(msg):
            pass

        adapter.on_message = handler
        assert adapter.on_message is handler

    @pytest.mark.asyncio
    async def test_edit_message_raises_not_implemented(self):
        adapter = _DummyAdapter()
        with pytest.raises(NotImplementedError, match="dummy"):
            await adapter.edit_message("ch-1", "msg-1", "new text")

    @pytest.mark.asyncio
    async def test_delete_message_raises_not_implemented(self):
        adapter = _DummyAdapter()
        with pytest.raises(NotImplementedError, match="dummy"):
            await adapter.delete_message("ch-1", "msg-1")
