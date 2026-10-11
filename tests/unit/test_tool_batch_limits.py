"""Reject oversized responses before executing any of their tools."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.litellm_adapter import _MAX_TOOL_CALLS_PER_RESPONSE, StreamResult
from tests.unit.test_live_streaming import _chunk, _make


def calls(count):
    return [SimpleNamespace(index=i, id=f"call_{i}", function=SimpleNamespace(
        name="file_write", arguments='{"path":"report.txt","content":"changed"}')) for i in range(count)]


@pytest.mark.parametrize("retry_overflows", [False, True])
async def test_oversized_batch_is_closed_and_never_executed(monkeypatch, retry_overflows):
    requests, closed, consumed = [], [], []

    async def completion(**kwargs):
        requests.append(kwargs)
        attempt = len(requests)

        async def chunks():
            try:
                if attempt == 1 or retry_overflows:
                    yield _chunk(content="Saving now")
                    for call in calls(_MAX_TOOL_CALLS_PER_RESPONSE + 100):
                        consumed.append(call.index)
                        yield _chunk(tool_calls=[call])
                else:
                    yield _chunk(content="Please use a smaller batch.", finish="stop")
            finally:
                closed.append(attempt)
        return chunks()

    monkeypatch.setattr("litellm.acompletion", completion)
    execute = AsyncMock()
    monkeypatch.setattr(StreamResult, "_execute_tool_batch", execute)
    stream = _make("openai/gpt-5.4")
    async for _ in stream.stream_text():
        pass
    assert len(requests) == 2 and closed == [1, 2]
    assert len(consumed) == (_MAX_TOOL_CALLS_PER_RESPONSE + 1) * (2 if retry_overflows else 1)
    execute.assert_not_called()
    assert not any(m.get("tool_calls") or m["role"] == "tool" for m in stream._messages)
    assert "None of those calls ran" in requests[1]["messages"][-1]["content"]
    assert stream.tool_budget_exhausted is retry_overflows
    assert "Saving now" not in await stream.get_output()


async def test_limit_counts_calls_not_argument_chunks(monkeypatch):
    responses = 0

    async def completion(**kwargs):
        nonlocal responses
        responses += 1

        async def chunks():
            if responses == 1:
                for call in calls(_MAX_TOOL_CALLS_PER_RESPONSE):
                    argument = call.function.arguments
                    yield _chunk(tool_calls=[SimpleNamespace(index=call.index, id=call.id,
                                 function=SimpleNamespace(name=call.function.name, arguments=argument[:10]))])
                    yield _chunk(tool_calls=[SimpleNamespace(index=call.index, id=None,
                                 function=SimpleNamespace(name=None, arguments=argument[10:]))])
                yield _chunk(finish="tool_calls")
            else:
                yield _chunk(content="Done.", finish="stop")
        return chunks()

    monkeypatch.setattr("litellm.acompletion", completion)
    execute = AsyncMock()
    monkeypatch.setattr(StreamResult, "_execute_tool_batch", execute)
    stream = _make("openai/gpt-5.4")
    async for _ in stream.stream_text():
        pass
    execute.assert_awaited_once()
    assert execute.call_args.args[0] == [{"id": call.id, "type": "function", "function": {
        "name": call.function.name, "arguments": call.function.arguments}} for call in calls(_MAX_TOOL_CALLS_PER_RESPONSE)]
    assert not stream.tool_budget_exhausted
