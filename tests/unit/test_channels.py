"""Tests for channel types and registry."""

from __future__ import annotations

import os
from unittest.mock import patch

import rune.channels.registry as registry_mod
from rune.channels.registry import ChannelRegistry
from rune.channels.types import (
    ChannelAdapter,
    IncomingMessage,
    OutgoingMessage,
)


class _FakeAdapter(ChannelAdapter):
    """Minimal concrete adapter for testing."""

    def __init__(self, adapter_name: str = "fake") -> None:
        super().__init__()
        self._name = adapter_name

    @property
    def name(self) -> str:
        return self._name

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


class TestIncomingMessage:
    def test_incoming_message_defaults(self):
        msg = IncomingMessage(channel_id="ch1", sender_id="user1", text="hello")
        assert msg.channel_id == "ch1"
        assert msg.sender_id == "user1"
        assert msg.text == "hello"
        assert msg.attachments == []
        assert msg.reply_to is None
        assert msg.metadata == {}


class TestChannelRegistry:
    def test_channel_registry_lifecycle(self):
        reg = ChannelRegistry()
        adapter = _FakeAdapter("test-channel")

        reg.register(adapter)
        assert reg.get("test-channel") is adapter
        assert "test-channel" in reg.list()

        reg.unregister("test-channel")
        assert reg.get("test-channel") is None
        assert "test-channel" not in reg.list()

    def test_auto_discover_no_env(self):
        """Without channel tokens in env, auto_discover returns empty list."""
        # Reset the singleton so we get a fresh registry
        old_registry = registry_mod._registry
        registry_mod._registry = None
        try:
            clean_env = {
                k: v for k, v in os.environ.items()
                if k not in (
                    "RUNE_TELEGRAM_TOKEN",
                    "RUNE_DISCORD_TOKEN",
                    "RUNE_SLACK_BOT_TOKEN",
                    "RUNE_SLACK_APP_TOKEN",
                )
            }
            with patch.dict(os.environ, clean_env, clear=True):
                discovered = registry_mod.auto_discover_channels()
                assert isinstance(discovered, list)
                assert len(discovered) == 0
        finally:
            registry_mod._registry = old_registry


class TestParseSlashCommand:
    def test_parse_slash_command_help(self):
        from rune.ui.commands import parse_slash_command

        result = parse_slash_command("/help")
        assert result is not None
        assert result[0] == "/help"
        assert result[1] == ""

    def test_parse_slash_command_with_args(self):
        from rune.ui.commands import parse_slash_command

        result = parse_slash_command("/model gpt-4")
        assert result is not None
        assert result[0] == "/model"
        assert result[1] == "gpt-4"

    def test_parse_non_command(self):
        from rune.ui.commands import parse_slash_command

        result = parse_slash_command("just a regular message")
        assert result is None
