"""A hosted owner connects an account from a card in the chat, and the task carries on with it."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.api.approvals import approval_callback
from rune.capabilities.connector import register_connector_capabilities
from rune.capabilities.registry import CapabilityRegistry
from rune.connectors.store import authority_key
from rune.safety.approval_context import parking_scope

GMAIL = {"name": "gmail", "origin": "https://gmail.googleapis.com", "path_prefix": "/gmail/v1/users/me/",
         "methods": ["GET"], "revision": "c" * 32}
NONE = {"connectors": []}
LISTED = {"connectors": [GMAIL]}
STATUS = {"closed": False, "untrusted": False, "approval_url": "https://approve.example.com/"}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://10.207.0.1:4100")
    authority_key(tmp_path / "broker", create=True)
    monkeypatch.setattr("rune.capabilities.connector.CONNECT_POLL_SECONDS", 0.01)


def broker(monkeypatch, lists, status=STATUS):
    """A broker that answers `list` with each of `lists` in turn (the last one from then on)."""
    calls = []

    async def call(operation, params):
        calls.append(operation)
        if operation == "status":
            return status
        assert operation == "list"
        return lists[min(calls.count("list"), len(lists)) - 1]

    monkeypatch.setattr("rune.capabilities.connector.call_broker", AsyncMock(side_effect=call))
    return calls


def tool(name, approve=None):
    registry = CapabilityRegistry()
    register_connector_capabilities(registry)
    opts = ToolAdapterOptions(approval_callback=approve, on_tool_end=AsyncMock())
    return _build_typed_tool(cap_def=registry.get(name), reg=registry, opts=opts, cache=None, stall=None).function


def channel(timeout_ms=5000):
    events, pending = [], {}

    async def emit(event, data):
        events.append((event, data))

    return approval_callback("run1", pending, emit, timeout_ms=timeout_ms), events, pending


async def shown(pending):
    async with asyncio.timeout(5):
        while not pending:
            await asyncio.sleep(0.001)


async def test_the_owner_connects_from_one_card_and_the_task_carries_on(hosted, monkeypatch):
    calls = broker(monkeypatch, [NONE, NONE, NONE, LISTED])
    approve, events, _ = channel()
    answer = await tool("connect_account", approve)(kind="gmail-read")
    assert json.loads(answer[answer.index("{"):])["connector"] == GMAIL, "the new connector, ready to call"
    cards = [d for e, d in events if e == "approval_request"]
    assert len(cards) == 1
    assert cards[0]["external"] == {"url": "https://approve.example.com/connect/gmail-read",
                                    "connect": {"label": "Connect Gmail",
                                                "detail": "Lets Rune read your mail. It can't send any."}}
    assert [e for e, _ in events][-1] == "approval_closed", "the card goes once it is connected"
    assert calls.count("list") == 4


async def test_an_account_already_connected_shows_no_card(hosted, monkeypatch):
    calls = broker(monkeypatch, [LISTED])
    approve, events, _ = channel()
    answer = await tool("connect_account", approve)(kind="gmail-read")
    assert '"connected": true' in answer
    assert events == [] and calls == ["list"]


async def test_not_now_ends_the_wait_and_the_agent_is_told_not_to_ask_again(hosted, monkeypatch):
    broker(monkeypatch, [NONE])
    approve, events, pending = channel()

    async def not_now():
        await shown(pending)
        next(iter(pending.values())).set_result({"decision": "deny"})

    presser = asyncio.create_task(not_now())
    answer = await tool("connect_account", approve)(kind="calendar-read")
    await presser
    assert "chose not to connect" in answer and "Don't ask again" in answer
    assert [e for e, _ in events] == ["approval_request", "approval_closed"]


async def test_a_card_left_alone_times_out(hosted, monkeypatch):
    broker(monkeypatch, [NONE])
    approve, _, _ = channel(timeout_ms=50)
    assert "didn't connect it in time" in await tool("connect_account", approve)(kind="gmail-send")


async def test_a_broker_that_blinks_while_the_owner_connects_is_asked_again(hosted, monkeypatch):
    answers = iter([NONE, httpx.ConnectError("down"), LISTED])

    async def call(operation, params):
        if operation == "status":
            return STATUS
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("rune.capabilities.connector.call_broker", AsyncMock(side_effect=call))
    approve, _, _ = channel()
    assert '"connected": true' in await tool("connect_account", approve)(kind="gmail-read")


async def test_without_a_card_to_show_the_link_is_given_to_pass_on(hosted, monkeypatch):
    broker(monkeypatch, [NONE])
    # A chat app's approvals (Telegram) have no card that waits.
    answer = await tool("connect_account", AsyncMock(return_value=False, spec=["__call__"]))(kind="gmail-read")
    assert "https://approve.example.com/connect/gmail-read" in answer and "try again" in answer


async def test_a_background_run_gives_the_link_rather_than_waiting(hosted, monkeypatch):
    broker(monkeypatch, [NONE])
    approve, events, _ = channel()
    with parking_scope(AsyncMock(return_value=True)):
        answer = await tool("connect_account", approve)(kind="gmail-read")
    assert "https://approve.example.com/connect/gmail-read" in answer
    assert events == []


async def test_no_approval_page_means_no_link(hosted, monkeypatch):
    broker(monkeypatch, [NONE], status={"closed": False, "approval_url": ""})
    approve, events, _ = channel()
    assert "no approval page" in await tool("connect_account", approve)(kind="gmail-read")
    assert events == []


async def test_a_local_install_connects_with_the_command(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNE_CONNECTOR_URL", raising=False)
    assert "rune connector add" in await tool("connect_account", channel()[0])(kind="gmail-read")


async def test_a_connector_the_broker_doesnt_have_is_reported_as_not_sent(hosted, monkeypatch):
    request = httpx.Request("POST", "http://10.207.0.1:4100/request")
    refused = httpx.HTTPStatusError("400", request=request, response=httpx.Response(400, request=request))
    monkeypatch.setattr("rune.capabilities.connector.call_broker", AsyncMock(side_effect=refused))
    answer = await tool("connector_request", channel()[0])(
        connector="gmail", origin="https://gmail.googleapis.com", revision="c" * 32, method="GET",
        path="/gmail/v1/users/me/messages")
    assert "nothing was sent" in answer and "connect_account" in answer
    assert "do not repeat" not in answer


async def test_other_broker_errors_still_count_as_uncertain(hosted, monkeypatch):
    request = httpx.Request("POST", "http://10.207.0.1:4100/request")
    failed = httpx.HTTPStatusError("502", request=request, response=httpx.Response(502, request=request))
    monkeypatch.setattr("rune.capabilities.connector.call_broker", AsyncMock(side_effect=failed))
    answer = await tool("connector_request", channel()[0])(
        connector="gmail", origin="https://gmail.googleapis.com", revision="c" * 32, method="POST",
        path="/gmail/v1/users/me/messages/send", body="{}")
    assert "do not repeat a write" in answer
