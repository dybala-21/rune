"""The daemon's inbound channel path: constructable, started, and reaching
the agent.

Two defects compounded here. ChannelGateway set `_conv_manager` in __init__
but omitted it from __slots__, so with no __dict__ the constructor raised
AttributeError — the gateway could not be built at all. The daemon caught
that in a bare except, so the symptom looked like "never started." And it
was in fact never started: the daemon opened the channel adapters but never
called gateway.start(), which is what points each adapter's on_message at
the gateway. Inbound Telegram/Discord/Slack messages hit `_on_message is
None` and were dropped, so channel use never fed the self-improving loop.
"""
from __future__ import annotations

import asyncio

import pytest

from rune.channels.registry import ChannelRegistry
from rune.channels.types import IncomingMessage
from rune.daemon.gateway import ChannelGateway


class _Adapter:
    def __init__(self) -> None:
        self.on_message = None
    @property
    def name(self) -> str:
        return "fake"
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, channel_id, message) -> None: ...
    async def send_notification(self, recipient, text) -> None: ...


class _Scheduler:
    def __init__(self) -> None:
        self.goals: list[str] = []
    async def execute(self, goal, sender_id):
        self.goals.append(goal)
        return "ok"


def test_gateway_is_constructable():
    # The __slots__ omission made this raise before it could do anything.
    ChannelGateway(ChannelRegistry())


@pytest.mark.asyncio
async def test_start_wires_and_stop_unwires_on_message():
    reg = ChannelRegistry()
    reg.register(_Adapter())
    a = reg.get("fake")
    gw = ChannelGateway(reg)
    assert a.on_message is None
    await gw.start()
    assert a.on_message is not None
    await gw.stop()
    assert a.on_message is None


@pytest.mark.asyncio
async def test_inbound_message_reaches_the_agent():
    reg = ChannelRegistry()
    reg.register(_Adapter())
    a = reg.get("fake")
    sched = _Scheduler()
    gw = ChannelGateway(reg, agent_scheduler=sched)
    await gw.start()
    try:
        await a.on_message(IncomingMessage(
            channel_id="c", sender_id="s", text="do the thing", attachments=[]))
        await asyncio.sleep(0.2)
        assert "do the thing" in sched.goals
    finally:
        await gw.stop()
