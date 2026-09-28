"""Exercise approval and caching at the tool dispatch boundary."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rune.agent.cognitive_cache import SessionToolCache
from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_HYBRID_API", "1")

    def build(name, approve=None, execute=None, cache=None):
        execute = execute or AsyncMock(return_value=CapabilityResult(success=True, output="sent"))
        finished = AsyncMock()
        cap = CapabilityDefinition(name=name, description=name, execute=execute)
        registry = CapabilityRegistry()
        registry.register(cap)
        tool = _build_typed_tool(
            cap_def=cap, reg=registry, cache=cache, stall=None,
            opts=ToolAdapterOptions(approval_callback=approve, on_tool_end=finished),
        )
        return tool.function, execute, finished

    return build


@pytest.mark.parametrize("second", [
    {"url": "https://mail.test/send", "method": "POST", "body": "to=alice"},
    {"url": "https://mail.test/send", "method": "POST", "body": "to=bob"},
    {"url": "https://mail.test/delete", "method": "POST", "body": "id=1"},
])
@pytest.mark.parametrize("cached", [False, True])
async def test_write_approval_is_not_reused(dispatch, second, cached):
    approve = AsyncMock(side_effect=[True, False])
    fetch, execute, finished = dispatch(
        "web_fetch", approve, cache=SessionToolCache() if cached else None,
    )
    await fetch(url="https://mail.test/send", method="POST", body="to=alice")
    answer = await fetch(**second)
    assert "User declined" in answer
    assert approve.await_count == 2
    execute.assert_awaited_once()
    assert finished.call_args.args[1].metadata["action_status"] == "not_executed"


async def test_cached_read_cannot_satisfy_a_write(dispatch):
    approve = AsyncMock(return_value=False)
    fetch, execute, _ = dispatch("web_fetch", approve, cache=SessionToolCache())
    await fetch(url="https://mail.test/message")
    answer = await fetch(url="https://mail.test/message", method="POST", body="reply=hello")
    assert "User declined" in answer
    approve.assert_awaited_once()
    execute.assert_awaited_once()
    await fetch(url="https://mail.test/message")
    execute.assert_awaited_once()


@pytest.mark.parametrize("write_success", [True, False])
async def test_read_after_write_checks_current_state(dispatch, write_success):
    state = {"sent": False}

    async def service(params):
        if params.get("method") == "POST":
            state["sent"] = True
            return CapabilityResult(success=write_success, output="accepted" if write_success else "",
                                    error=None if write_success else "response lost")
        return CapabilityResult(success=True, output="sent" if state["sent"] else "draft")

    execute = AsyncMock(side_effect=service)
    fetch, _, _ = dispatch("web_fetch", AsyncMock(return_value=True), execute, SessionToolCache())
    assert "draft" in await fetch(url="https://mail.test/message/1")
    await fetch(url="https://mail.test/send", method="POST", body="id=1")
    assert "sent" in await fetch(url="https://mail.test/message/1")
    assert execute.await_count == 3


async def test_cancelled_write_invalidates_pages_but_keeps_file_reads(dispatch):
    cache = SessionToolCache()
    params = {"path": "/fixture/source.py"}
    key = cache.generate_key("file_read", params)
    cache.set(key, "file_read", params, "source", step_number=0)
    execute = AsyncMock(side_effect=[
        CapabilityResult(success=True, output="draft"),
        asyncio.CancelledError(),
        CapabilityResult(success=True, output="sent"),
    ])
    fetch, _, _ = dispatch("web_fetch", AsyncMock(return_value=True), execute, cache)
    await fetch(url="https://mail.test/message/1")
    with pytest.raises(asyncio.CancelledError):
        await fetch(url="https://mail.test/send", method="POST")
    assert "sent" in await fetch(url="https://mail.test/message/1")
    assert cache.get(key, "file_read", params) is not None


@pytest.mark.parametrize("name,params", [
    ("browser_act", {"action": "click", "selector": "e1"}),
    ("browser_batch", {"actions": [{"action": "click", "selector": "e1"}]}),
    ("browser_workflow", {"steps": [{"action": "click", "selector": "e1"}]}),
    ("mcp.mail.send_message", {"to": "alice@example.test", "body": "hello"}),
])
async def test_strict_actions_require_an_approval_channel(dispatch, monkeypatch, name, params):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "strict")
    run, execute, finished = dispatch(name)
    assert "no approval channel" in (await run(**params)).lower()
    execute.assert_not_awaited()
    assert finished.call_args.args[1].metadata["action_status"] == "not_executed"


async def test_strict_browser_click_approval_is_not_reused(dispatch, monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "strict")
    approve = AsyncMock(side_effect=[True, False])
    click, execute, _ = dispatch("browser_act", approve)
    await click(action="click", selector="e1")
    assert "User declined" in await click(action="click", selector="e2")
    assert approve.await_count == 2
    execute.assert_awaited_once()


@pytest.mark.parametrize("mode,allowed", [("standard", False), ("bypass", True)])
async def test_headless_mcp_write_honors_approval_mode(dispatch, monkeypatch, mode, allowed):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", mode)
    send, execute, _ = dispatch("mcp.mail.send_message")
    await send(to="alice@example.test", body="hello")
    assert execute.await_count == int(allowed)


async def test_strict_read_approval_stays_reusable(dispatch, monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "strict")
    approve = AsyncMock(return_value=True)
    fetch, execute, _ = dispatch("web_fetch", approve)
    await fetch(url="https://mail.test/message/1")
    await fetch(url="https://mail.test/message/2")
    approve.assert_awaited_once()
    assert execute.await_count == 2


async def test_headless_mcp_read_needs_no_approval(dispatch):
    read, execute, _ = dispatch("mcp.mail.list_messages")
    await read()
    execute.assert_awaited_once()
