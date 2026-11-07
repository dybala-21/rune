"""Hosted workers leave connector approval to the broker and wait for the owner's decision there."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.api.approvals import approval_callback
from rune.capabilities.connector import register_connector_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.connectors.store import authority_key

PENDING = {"success": False, "error": "Waiting for the owner", "metadata": {
    "action_status": "not_executed", "approval_status": "pending", "approval_id": "a" * 32,
    "approval_url": "https://approve.example.com/"}}
RAN = {"success": True, "output": "sent", "metadata": {"status_code": 200}}
PARAMS = {"connector": "mail", "origin": "https://mail.test", "revision": "b" * 32, "method": "POST",
          "path": "/send", "body": "{\"to\":\"x\"}"}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    monkeypatch.setattr("rune.capabilities.connector.APPROVAL_POLL_SECONDS", 0.01)


def tool(approve):
    registry = CapabilityRegistry()
    register_connector_capabilities(registry)
    opts = ToolAdapterOptions(approval_callback=approve, on_tool_end=AsyncMock())
    return _build_typed_tool(cap_def=registry.get("connector_request"), reg=registry, opts=opts,
                             cache=None, stall=None).function


async def opened(pending):
    """Until the app shows an approval card; a card that never appears fails the test, not hangs it."""
    async with asyncio.timeout(5):
        while not pending:
            await asyncio.sleep(0.001)


def channel(timeout_ms=5000):
    events, pending = [], {}

    async def emit(event, data):
        events.append((event, data))

    return approval_callback("run1", pending, emit, timeout_ms=timeout_ms), events, pending


async def test_the_app_only_shows_the_approval_page_and_the_call_runs_once_the_owner_approves(hosted, monkeypatch):
    broker = AsyncMock(side_effect=[PENDING, PENDING, PENDING, RAN])
    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    approve, events, pending = channel()

    async def press_approve_in_the_app():
        await opened(pending)
        # Approving inside the app (the VM) must not count: only the approval page does.
        next(iter(pending.values())).set_result({"decision": "approve_once"})

    presser = asyncio.create_task(press_approve_in_the_app())
    answer = await tool(approve)(**PARAMS)
    await presser
    assert "sent" in answer
    assert broker.await_count == 4, "re-sent until the broker stopped saying pending"
    shown = [d for e, d in events if e == "approval_request"]
    assert len(shown) == 1 and shown[0]["external"] == {"url": "https://approve.example.com/"}
    assert [e for e, _ in events][-1] == "approval_closed"


async def test_declining_in_the_app_stops_waiting_and_nothing_runs(hosted, monkeypatch):
    broker = AsyncMock(return_value=PENDING)
    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    approve, events, pending = channel()

    async def decline():
        await opened(pending)
        next(iter(pending.values())).set_result({"decision": "deny"})

    decliner = asyncio.create_task(decline())
    answer = await tool(approve)(**PARAMS)
    await decliner
    assert "declined" in answer.lower()
    calls = broker.await_count
    await asyncio.sleep(0.05)
    assert broker.await_count == calls, "no request is sent after the user declined"


async def test_an_unanswered_approval_times_out_without_running(hosted, monkeypatch):
    monkeypatch.setattr("rune.capabilities.connector.call_broker", AsyncMock(return_value=PENDING))
    approve, _, _ = channel(timeout_ms=50)
    assert "timed out" in (await tool(approve)(**PARAMS)).lower()


async def test_without_an_approval_channel_the_page_is_named_and_nothing_runs(hosted, monkeypatch):
    broker = AsyncMock(return_value=PENDING)
    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    answer = await tool(None)(**PARAMS)
    assert "Waiting for the owner" in answer
    assert broker.await_count == 1


async def test_a_local_install_still_asks_in_the_app(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.delenv("RUNE_CONNECTOR_URL", raising=False)
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    broker = AsyncMock(return_value=RAN)
    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    approve = AsyncMock(return_value=False)
    answer = await tool(approve)(**PARAMS)
    approve.assert_awaited_once()
    broker.assert_not_awaited()
    assert "declined" in answer.lower()
