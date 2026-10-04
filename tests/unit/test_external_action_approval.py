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


async def test_authenticated_connector_reads_each_need_their_own_approval(dispatch):
    approve = AsyncMock(side_effect=[True, False])
    fetch, execute, _ = dispatch("connector_request", approve)
    params = {"connector": "mail", "origin": "https://mail.test", "path": "/inbox", "method": "GET"}
    await fetch(**params)
    assert "User declined" in await fetch(**params)
    assert approve.await_count == 2
    execute.assert_awaited_once()


async def test_authenticated_connector_reads_fail_closed_without_a_channel(dispatch):
    fetch, execute, _ = dispatch("connector_request")
    assert "no approval channel" in (await fetch(connector="mail", origin="https://mail.test", path="/inbox")).lower()
    execute.assert_not_awaited()


async def test_headless_mcp_read_needs_no_approval(dispatch):
    read, execute, _ = dispatch("mcp.mail.list_messages")
    await read()
    execute.assert_awaited_once()


async def test_direct_dispatch_cannot_bypass_or_reuse_an_approval(monkeypatch):
    from rune.safety.approval_context import approval_granted

    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    monkeypatch.setenv("RUNE_HYBRID_API", "1")
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="web_fetch", description="fetch", execute=execute))
    params = {"url": "https://mail.test/send", "method": "POST", "body": "to=alice"}
    with approval_granted():
        denied = await registry.execute("web_fetch", params)
    assert denied.metadata["requires_approval"]
    execute.assert_not_awaited()

    with approval_granted("web_fetch", params):
        changed = await registry.execute("web_fetch", {**params, "body": "to=bob"})
        assert not changed.success
        assert (await registry.execute("web_fetch", params)).success
        repeated = await registry.execute("web_fetch", params)
        assert not repeated.success
    execute.assert_awaited_once()


async def test_parallel_calls_cannot_spend_the_same_grant_twice(monkeypatch):
    from rune.safety.approval_context import approval_granted

    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    registry = CapabilityRegistry()
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    registry.register(CapabilityDefinition(name="mcp.mail.send", description="send", execute=execute))
    params = {"recipient": "alice"}
    with approval_granted("mcp.mail.send", params):
        results = await asyncio.gather(*(registry.execute("mcp.mail.send", params) for _ in range(2)))
    assert sum(result.success for result in results) == 1
    execute.assert_awaited_once()


async def test_grant_does_not_authorize_nested_capability_calls(monkeypatch):
    from rune.safety.approval_context import approval_granted

    monkeypatch.setenv("RUNE_APPROVAL_MODE", "standard")
    registry = CapabilityRegistry()
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    registry.register(CapabilityDefinition(name="mcp.mail.send", description="send", execute=execute))

    async def outer(params):
        return await registry.execute("mcp.mail.send", {"recipient": "bob"})

    registry.register(CapabilityDefinition(name="mcp.calendar.create", description="create", execute=outer))
    with approval_granted("mcp.calendar.create", {}):
        result = await registry.execute("mcp.calendar.create", {})
    assert not result.success
    execute.assert_not_awaited()


async def test_parameters_cannot_change_while_approval_is_pending(dispatch):
    body = {"recipient": "alice"}

    async def approve(*args):
        body["recipient"] = "bob"
        return True

    fetch, execute, _ = dispatch("web_fetch", approve)
    await fetch(url="https://mail.test/send", method="POST", body=body)
    execute.assert_not_awaited()


async def test_direct_dispatch_blocks_when_guardian_fails(monkeypatch):
    monkeypatch.setattr("rune.safety.guardian.get_guardian", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="file_write", description="write", execute=execute))
    result = await registry.execute("file_write", {"path": "/tmp/report", "content": "report"})
    assert result.metadata["action_status"] == "not_executed"
    execute.assert_not_awaited()


async def test_invalid_arguments_return_feedback_without_requesting_approval():
    from pydantic import BaseModel

    class Params(BaseModel):
        recipient: str

    execute = AsyncMock(return_value=CapabilityResult(success=True))
    approve = AsyncMock(return_value=True)
    finished = AsyncMock()
    registry = CapabilityRegistry()
    cap = CapabilityDefinition(name="mcp.mail.send", description="send", execute=execute, parameters_model=Params)
    registry.register(cap)
    wrapped = _build_typed_tool(cap_def=cap, reg=registry, cache=None, stall=None,
                               opts=ToolAdapterOptions(approval_callback=approve, on_tool_end=finished))
    assert "recipient: Field required" in await wrapped.function()
    approve.assert_not_awaited()
    execute.assert_not_awaited()
    assert finished.call_args.args[1].metadata["action_status"] == "not_executed"
