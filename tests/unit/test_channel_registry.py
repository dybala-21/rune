"""Tests for the channel registry module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rune.channels.registry import ChannelRegistry, auto_discover_channels
from rune.channels.types import ChannelAdapter, OutgoingMessage

# ---------------------------------------------------------------------------
# Stub adapter for testing
# ---------------------------------------------------------------------------

class _StubAdapter(ChannelAdapter):
    """Minimal concrete adapter for testing."""

    def __init__(self, adapter_name: str = "stub") -> None:
        super().__init__()
        self._name = adapter_name
        self.started = False
        self.stopped = False

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, channel_id: str, message: OutgoingMessage) -> str | None:
        pass

    async def send_approval(self, channel_id: str, description: str, approval_id: str) -> None:
        pass

    async def ask_question(self, channel_id: str, question: str, options: list[str] | None = None, timeout: float = 300.0) -> str:
        return ""


class _FailingAdapter(_StubAdapter):
    async def start(self) -> None:
        raise RuntimeError("connection failed")

    async def stop(self) -> None:
        raise RuntimeError("disconnect failed")


class TestChannelRegistry:
    def test_register_and_get(self):
        reg = ChannelRegistry()
        adapter = _StubAdapter("test-channel")
        reg.register(adapter)
        assert reg.get("test-channel") is adapter

    def test_get_missing_returns_none(self):
        reg = ChannelRegistry()
        assert reg.get("nonexistent") is None

    def test_list_channels(self):
        reg = ChannelRegistry()
        reg.register(_StubAdapter("alpha"))
        reg.register(_StubAdapter("beta"))
        names = reg.list()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_unregister(self):
        reg = ChannelRegistry()
        reg.register(_StubAdapter("to-remove"))
        assert reg.get("to-remove") is not None
        reg.unregister("to-remove")
        assert reg.get("to-remove") is None

    def test_unregister_missing_no_error(self):
        reg = ChannelRegistry()
        # Should not raise
        reg.unregister("nonexistent")

    def test_register_duplicate_overwrites(self):
        reg = ChannelRegistry()
        a1 = _StubAdapter("dup")
        a2 = _StubAdapter("dup")
        reg.register(a1)
        reg.register(a2)
        assert reg.get("dup") is a2

    @pytest.mark.asyncio
    async def test_start_all(self):
        reg = ChannelRegistry()
        a1 = _StubAdapter("a")
        a2 = _StubAdapter("b")
        reg.register(a1)
        reg.register(a2)
        await reg.start_all()
        assert a1.started is True
        assert a2.started is True

    @pytest.mark.asyncio
    async def test_stop_all(self):
        reg = ChannelRegistry()
        a1 = _StubAdapter("a")
        a2 = _StubAdapter("b")
        reg.register(a1)
        reg.register(a2)
        await reg.stop_all()
        assert a1.stopped is True
        assert a2.stopped is True

    @pytest.mark.asyncio
    async def test_start_all_handles_failure(self):
        """A failing adapter should not prevent others from starting."""
        reg = ChannelRegistry()
        failing = _FailingAdapter("bad")
        good = _StubAdapter("good")
        reg.register(failing)
        reg.register(good)
        # Should not raise
        await reg.start_all()
        assert good.started is True

    @pytest.mark.asyncio
    async def test_stop_all_handles_failure(self):
        """A failing adapter should not prevent others from stopping."""
        reg = ChannelRegistry()
        failing = _FailingAdapter("bad")
        good = _StubAdapter("good")
        reg.register(failing)
        reg.register(good)
        await reg.stop_all()
        assert good.stopped is True


class TestAutoDiscover:
    def test_no_env_vars_discovers_nothing(self):
        with patch.dict("os.environ", {}, clear=True):
            # Clear any existing singleton
            import rune.channels.registry as mod
            old = mod._registry
            mod._registry = None
            try:
                discovered = auto_discover_channels()
                assert discovered == []
            finally:
                mod._registry = old
