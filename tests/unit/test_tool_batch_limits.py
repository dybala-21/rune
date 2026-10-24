"""Reject oversized responses before executing any of their tools."""

import json
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


async def test_sequential_form_edits_do_not_trip_the_duplicate_call_policy(monkeypatch):
    executed = []

    async def execute(name, params):
        executed.append((name, dict(params)))
        params["checks"] = [{"selector": "#receipt"}]
        return "Action dispatched."

    stream = _make("openai/gpt-5.4")
    monkeypatch.setattr(stream, "_execute_tool", execute)
    for index in range(10):
        await stream._execute_tool_batch([{"id": str(index), "function": {
            "name": "browser_act", "arguments": json.dumps({"action": "type", "selector": f"#field{index}", "value": "3"}),
        }}])
    assert len(executed) == 10
    args = {"action": "click", "selector": "#save"}
    for index in range(8):
        await stream._execute_tool_batch([{"id": f"save{index}", "function": {
            "name": "browser_act", "arguments": json.dumps(args),
        }}])
    assert len(executed) == 17
    assert any("repeated the same arguments" in str(message.get("content")) for message in stream._messages)


async def test_added_batch_checks_do_not_change_the_original_call_identity(monkeypatch):
    executed = []

    async def execute(name, params):
        executed.append(name)
        params["actions"][0]["params"]["checks"] = [{"selector": "#receipt"}]
        return "Action dispatched."

    stream = _make("openai/gpt-5.4")
    monkeypatch.setattr(stream, "_execute_tool", execute)
    args = {"actions": [{"type": "act", "params": {"action": "click", "selector": "#save"}}]}
    for index in range(8):
        await stream._execute_tool_batch([{"id": str(index), "function": {
            "name": "browser_batch", "arguments": json.dumps(args),
        }}])
    assert len(executed) == 7


@pytest.mark.parametrize('model', ['openai/gpt-6-astra', 'xai/grok-4.7', 'vertex_ai/gemini-2.5-flash', 'anthropic/claude-opus-5'])
@pytest.mark.parametrize('recover', [False, True])
async def test_rejected_calls_stop_without_spending_another_model_request(monkeypatch, model, recover):
    from rune.agent.tool_output import RejectedToolResult
    from tests.unit.test_live_streaming import _tc

    requests, executed = [], []

    async def tool(**params):
        executed.append(params)
        return 'Action completed.'

    async def completion(**kwargs):
        requests.append(kwargs)
        attempt = len(requests)

        async def chunks():
            if recover and attempt == 4:
                yield _chunk(content='Done.', finish='stop')
            else:
                args = '{}' if recover and attempt == 3 else '[]'
                yield _chunk(tool_calls=_tc('noop', args))
                yield _chunk(finish='tool_calls')
        return chunks()

    monkeypatch.setattr('litellm.acompletion', completion)
    stream = _make(model)
    stream._tool_lookup['noop'] = tool
    async for _ in stream.stream_text():
        pass
    assert len(requests) == (4 if recover else 3)
    assert len(executed) == int(recover)
    assert bool(stream._task_blocked) is not recover
    if recover:
        assert await stream.get_output() == 'Done.'
    else:
        assert 'invalid_arguments' in await stream.get_output()
    assert not stream.tool_budget_exhausted
    assert all(isinstance(m['content'], str) for m in stream._messages if m['role'] == 'tool')
    assert isinstance(await stream._execute_tool('missing', {}), RejectedToolResult)


def test_executed_failures_do_not_count_as_rejected_calls():
    from rune.agent.tool_call_policy import ToolCallPolicy
    from rune.agent.tool_output import RejectedToolResult

    policy = ToolCallPolicy()
    rejected = RejectedToolResult('[ERROR] Invalid arguments', 'invalid_arguments')
    for _ in range(10):
        assert policy.record_result('file_edit', rejected) is None
        assert policy.record_result('bash_execute', '[ERROR] Two tests failed') is None
    assert policy.record_result('file_edit', rejected) is None
    assert policy.record_result('browser_act', rejected) is None
    assert policy.record_result('browser_batch', rejected)
    policy.reset()
    assert policy.record_result('file_edit', rejected) is None


async def test_parallel_read_progress_resets_rejection_streak(monkeypatch):
    from rune.agent.tool_output import RejectedToolResult

    async def execute(name, params):
        return 'Found report.csv' if name == 'file_list' else RejectedToolResult('[ERROR] Invalid path argument', 'invalid_arguments')

    stream = _make('openai/gpt-6-astra')
    monkeypatch.setattr(stream, '_execute_tool', execute)
    def batch(names):
        return [{'id': str(i), 'function': {'name': name, 'arguments': '{}'}} for i, name in enumerate(names)]
    for _ in range(5):
        await stream._execute_tool_batch(batch(['file_read', 'file_search', 'file_list']))
        assert not getattr(stream, '_task_blocked', '')
    await stream._execute_tool_batch(batch(['file_read', 'file_search', 'code_analyze']))
    assert not getattr(stream, '_task_blocked', '')


@pytest.mark.parametrize('arguments', ['[]', '{"path":'])
@pytest.mark.parametrize('names', [['bash_execute'], ['file_edit'], ['file_read', 'file_list']])
async def test_malformed_arguments_never_dispatch_or_crash_bookkeeping(arguments, names):
    stream = _make('openai/gpt-6-astra')
    stream._guided = True
    execute = AsyncMock(return_value='Done.')
    stream._tool_lookup.update({name: execute for name in names})
    for attempt in range(3):
        await stream._execute_tool_batch([{'id': f'{attempt}-{index}', 'function': {
            'name': name, 'arguments': arguments,
        }} for index, name in enumerate(names)])
    execute.assert_not_awaited()
    assert 'invalid_arguments' in stream._task_blocked
    assert all('JSON object' in message['content'] for message in stream._messages if message['role'] == 'tool')
