"""Usage and tool handling for incomplete Responses streams."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rune.agent.litellm_adapter import StreamResult, _litellm
from rune.llm.usage import token_counts


def terminal(status, reason=None, usage=True):
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator as Bridge,
    )
    _litellm()
    return Bridge.translate_responses_chunk_to_openai_stream({
        "type": f"response.{status}", "response": {"status": status, "output": [],
        "incomplete_details": {"reason": reason} if reason else None,
        "usage": {"input_tokens": 6660, "output_tokens": 573, "total_tokens": 7233,
                  "input_tokens_details": {"cached_tokens": 5868, "cache_write_tokens": 789},
                  "output_tokens_details": {"reasoning_tokens": 10}} if usage else None}})


@pytest.mark.parametrize("status,reason,finish", [
    ("completed", None, "stop"), ("incomplete", "max_output_tokens", "length"),
    ("incomplete", "content_filter", "content_filter"), ("failed", None, "error"),
])
def test_terminal_usage_and_finish_reason(status, reason, finish):
    chunk = terminal(status, reason)
    assert chunk.choices[0].finish_reason == finish
    counts = token_counts(chunk.usage)
    assert counts["total_tokens"] == 7233
    assert counts["cache_write_reported"] and counts["cache_write_tokens"] == 789
    assert counts["cached_input_tokens"] == 5868 and counts["reasoning_tokens"] == 10


def test_missing_usage_is_not_fabricated():
    chunk = terminal("incomplete", "max_output_tokens", usage=False)
    assert token_counts(getattr(chunk, "usage", None)) is None


@pytest.mark.parametrize("status,reason", [("failed", None), ("incomplete", "content_filter")])
async def test_rejected_response_does_not_execute_partial_tools(status, reason):
    sr = StreamResult(model="gpt-6-astra", messages=[], tool_schemas=[], tool_lookup={},
                      max_tokens=1000, temperature=0, request_tokens_limit=100_000, response_tokens_limit=1000)
    async def complete(**kwargs):
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(
                content=None, tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(
                    name="file_write", arguments='{"path":"output.txt","content":"partial"}'))]))], usage=None)
            yield terminal(status, reason)
        return stream()
    with patch("litellm.acompletion", new=complete), patch.object(sr, "_execute_tool_batch", new=AsyncMock()) as execute:
        with pytest.raises(RuntimeError, match="response ended"):
            async for _ in sr.stream_text():
                pass
        execute.assert_not_awaited()
    assert sr.usage().input_tokens + sr.usage().output_tokens == 7233


@pytest.mark.parametrize("args", ['{"path":"output.txt","content":"partial"}', '{"path":'])
async def test_length_limit_never_executes_incomplete_batch(args):
    sr = StreamResult(model="gpt-6-astra", messages=[], tool_schemas=[], tool_lookup={},
                      max_tokens=1000, temperature=0, request_tokens_limit=100_000, response_tokens_limit=1000)
    calls = 0
    async def complete(**kwargs):
        nonlocal calls
        calls += 1
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(
                content=None, tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(
                    name="file_write", arguments=args))]))], usage=None)
            yield terminal("incomplete", "max_output_tokens")
        return stream()
    with patch("litellm.acompletion", new=complete), patch.object(sr, "_execute_tool_batch", new=AsyncMock()) as execute:
        async for _ in sr.stream_text():
            pass
        execute.assert_not_awaited()
    assert sr.tool_budget_exhausted and calls == 3
    assert sr.usage().output_tokens == calls * 573


async def test_terminal_usage_settles_the_budget_once():
    from rune.agent.request_budget import check_request_budget
    from rune.agent.timing import capture_timing, timed_completion, timing_snapshot

    async def complete(**kwargs):
        async def stream():
            yield terminal("incomplete", "max_output_tokens")
        return stream()
    with capture_timing() as run:
        stream = await timed_completion(complete, {"model": "gpt-6-astra", "stream": True})
        chunks = [chunk async for chunk in stream]
        await stream.aclose()
        check_request_budget(run)
    counts = timing_snapshot(run)["usage"]
    assert chunks[0].choices[0].finish_reason == "length"
    assert counts["total_tokens"] == 7233 and counts["reported_calls"] == 1
    assert counts["unpriced_calls"] == 0 and counts["cost_usd"] > 0


def test_terminal_bridge_accepts_sdk_models_and_is_installed_once():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator as Bridge,
    )
    from openai.types.responses import ResponseIncompleteEvent

    from rune.llm.usage import preserve_responses_terminal_events

    preserve_responses_terminal_events()
    original = Bridge.translate_responses_chunk_to_openai_stream
    preserve_responses_terminal_events()
    assert original is Bridge.translate_responses_chunk_to_openai_stream
    event = ResponseIncompleteEvent.model_construct(type="response.incomplete", response={
        "output": [], "incomplete_details": {"reason": "max_output_tokens"}, "usage": None})
    assert original(event).choices[0].finish_reason == "length"
