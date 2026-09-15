import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rune.agent.attachments import content_text
from rune.agent.litellm_adapter import StreamResult
from rune.agent.run_control import ControlChanged, RunControl, control_scope, dispatch_scope
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult


async def test_pause_drains_dispatched_work_but_rejects_queued_work_after_resume():
    control = RunControl("run")
    started, release = asyncio.Event(), asyncio.Event()
    writes = []

    async def first():
        async with dispatch_scope():
            started.set()
            await release.wait()
            writes.append("first")

    with control_scope(control):
        task = asyncio.create_task(first())
        await started.wait()
        control.pause()
        assert control.state == "pausing"
        with pytest.raises(ControlChanged):
            async with dispatch_scope():
                writes.append("queued")
        release.set()
        await task
        assert control.state == "paused" and writes == ["first"]
        control.resume("Use the revised total")
        with pytest.raises(ControlChanged):
            async with dispatch_scope():
                writes.append("old plan")
        messages = []
        assert await control.checkpoint(messages)
        async with dispatch_scope():
            writes.append("new plan")
        assert messages[-1]["content"] == "Use the revised total"
        assert writes == ["first", "new plan"]


async def test_registry_rechecks_dispatch_after_an_approval_or_other_wait():
    control = RunControl("run")
    writes = []

    async def write(_):
        writes.append(1)
        return CapabilityResult(success=True)

    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="test_write", description="test", execute=write))
    with control_scope(control):
        control.pause()
        control.resume()
        result = await registry.execute("test_write", {})
        assert not result.success and "Not executed" in result.error
        assert not writes


async def test_stop_wakes_a_paused_loop_without_dispatching():
    control = RunControl("run")
    control.pause()
    task = asyncio.create_task(control.checkpoint([]))
    await asyncio.sleep(0)
    assert not task.done()
    control.stop()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("model", ["gpt-6-astra", "claude-opus-5"])
async def test_response_from_before_takeover_is_discarded_and_update_reaches_next_request(model):
    control = RunControl("run")
    requests, executed = [], []
    first_streamed, finish_stream = asyncio.Event(), asyncio.Event()

    def chunk(*, content=None, calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=calls), finish_reason="stop",
        )], usage=None)

    async def completion(**kwargs):
        requests.append([dict(message) for message in kwargs["messages"]])
        count = len(requests)

        async def stream():
            if count == 1:
                first_streamed.set()
                await finish_stream.wait()
                yield chunk(calls=[SimpleNamespace(index=0, id="old-call", function=SimpleNamespace(
                    name="write", arguments="{}"))])
            else:
                yield chunk(content="Updated task completed.")
        return stream()

    async def write(**_):
        executed.append(True)
        return "done"

    sr = StreamResult(model=model, messages=[{"role": "user", "content": "Review the expense"}],
                      tool_schemas=[{"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}}],
                      tool_lookup={"write": write}, max_tokens=200, temperature=0,
                      request_tokens_limit=100_000, response_tokens_limit=200)

    async def drive():
        with control_scope(control):
            return [text async for text in sr.stream_text()]

    with patch("litellm.acompletion", new=completion):
        task = asyncio.create_task(drive())
        await first_streamed.wait()
        control.pause()
        finish_stream.set()
        await asyncio.sleep(0)
        assert not task.done()
        control.resume("Exclude the old expense")
        await task
    assert not executed and len(requests) == 2
    assert content_text(requests[-1][-1]["content"]) == "Exclude the old expense"
    assert not any(message.get("tool_calls") for message in sr._messages)
    assert await sr.get_output() == "Updated task completed."
