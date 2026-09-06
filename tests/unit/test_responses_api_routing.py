"""Newer OpenAI models cannot take tools on /v1/chat/completions at all.

Measured live: gpt-6-astra and the gpt-5.6 family reject `tools` there with or
without an explicit reasoning_effort, since they carry a default one and
"none" is refused; gpt-5.3-codex is not served on that endpoint. All of them
handle tools on /v1/responses.

An agent step always carries tools, so those models could not take a single
step. Dropping the parameter removed the wasted retries but never made the
call work.
"""

from __future__ import annotations

import pytest

from rune.agent.model_traits import (
    is_responses_only_error,
    needs_responses_api,
    note_responses_only,
)

CHAT_TOOLS_REFUSED = (
    "litellm.BadRequestError: OpenAIException - Function tools with "
    "reasoning_effort are not supported for gpt-6-astra in "
    "/v1/chat/completions. To use function tools, use /v1/responses "
    "or set reasoning_effort to none."
)
NOT_ON_THIS_ENDPOINT = (
    "litellm.BadRequestError: OpenAIException - This model is not supported "
    "in the v1/chat/completions endpoint. Try v1/responses instead."
)


@pytest.fixture(autouse=True)
def _clear():
    from rune.agent import model_traits as mt

    mt._RESPONSES_ONLY.clear()
    yield
    mt._RESPONSES_ONLY.clear()


@pytest.mark.parametrize("message", [CHAT_TOOLS_REFUSED, NOT_ON_THIS_ENDPOINT])
def test_the_endpoint_refusals_are_recognised(message):
    assert is_responses_only_error(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "temperature is not supported with this model",
        "429 rate limit exceeded",
        "context_length_exceeded",
        "",
    ],
)
def test_unrelated_errors_are_not_claimed(message):
    assert is_responses_only_error(Exception(message)) is False


def test_a_model_is_remembered_and_others_are_not():
    assert needs_responses_api("gpt-6-astra") is False

    note_responses_only("gpt-6-astra")

    assert needs_responses_api("gpt-6-astra") is True
    assert needs_responses_api("gpt-5.4") is False


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def test_chat_tools_become_responses_tools():
    from rune.agent.responses_bridge import to_responses_tools

    out = to_responses_tools([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ])

    assert out == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]


def test_messages_become_responses_input():
    from rune.agent.responses_bridge import to_responses_input

    out = to_responses_input([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])

    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[1]["content"] == "hello"


def test_tool_results_are_carried_across():
    """A tool result must reach the model, or the loop forgets what it ran."""
    from rune.agent.responses_bridge import to_responses_input

    out = to_responses_input([
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ])

    kinds = [m.get("type") or m.get("role") for m in out]
    assert "function_call" in kinds
    assert "function_call_output" in kinds
    call = next(m for m in out if m.get("type") == "function_call")
    assert call["call_id"] == "call_1"
    assert call["name"] == "get_weather"
    result = next(m for m in out if m.get("type") == "function_call_output")
    assert result["call_id"] == "call_1"
    assert result["output"] == "sunny"


# ---------------------------------------------------------------------------
# Stream translation. Event shapes below are copied from a live gpt-6-astra
# run, not invented.
# ---------------------------------------------------------------------------

class _Event(dict):
    """Responses events arrive as objects; dict access is what the bridge uses."""


async def _drain(events):
    from rune.agent.responses_bridge import ResponsesToChatStream

    class _Src:
        def __aiter__(self):
            self._i = iter(events)
            return self

        async def __anext__(self):
            try:
                return next(self._i)
            except StopIteration:
                raise StopAsyncIteration from None

    return [c async for c in ResponsesToChatStream(_Src())]


@pytest.mark.asyncio
async def test_text_deltas_become_content():
    chunks = await _drain([
        _Event(type="ResponsesAPIStreamEvents.RESPONSE_CREATED", response={}),
        _Event(type="ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA", delta="Hel"),
        _Event(type="ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA", delta="lo"),
        _Event(type="ResponsesAPIStreamEvents.RESPONSE_COMPLETED",
               response={"output": [], "usage": {"input_tokens": 5, "output_tokens": 2}}),
    ])

    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage["output_tokens"] == 2


@pytest.mark.asyncio
async def test_a_function_call_arrives_as_tool_call_deltas():
    chunks = await _drain([
        _Event(type="ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED", output_index=0,
               item={"type": "function_call", "call_id": "call_1", "name": "get_weather"}),
        _Event(type="ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA",
               output_index=0, delta='{"city"'),
        _Event(type="ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA",
               output_index=0, delta=':"Seoul"}'),
        _Event(type="ResponsesAPIStreamEvents.RESPONSE_COMPLETED",
               response={"output": [{"type": "function_call"}], "usage": {}}),
    ])

    # Accumulate exactly as the streaming loop does.
    by_index: dict[int, dict] = {}
    for chunk in chunks:
        for choice in chunk.choices:
            for call in choice.delta.tool_calls or []:
                entry = by_index.setdefault(
                    call.index, {"id": "", "name": "", "arguments": ""}
                )
                if call.id:
                    entry["id"] = call.id
                if call.function.name:
                    entry["name"] = call.function.name
                entry["arguments"] += call.function.arguments

    assert by_index == {
        0: {"id": "call_1", "name": "get_weather", "arguments": '{"city":"Seoul"}'}
    }
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_two_parallel_calls_stay_separate():
    chunks = await _drain([
        _Event(type="OUTPUT_ITEM_ADDED", output_index=0,
               item={"type": "function_call", "call_id": "a", "name": "one"}),
        _Event(type="OUTPUT_ITEM_ADDED", output_index=1,
               item={"type": "function_call", "call_id": "b", "name": "two"}),
        _Event(type="FUNCTION_CALL_ARGUMENTS_DELTA", output_index=1, delta="{}"),
        _Event(type="RESPONSE_COMPLETED",
               response={"output": [{"type": "function_call"}], "usage": {}}),
    ])

    indexes = {
        call.index
        for chunk in chunks
        for choice in chunk.choices
        for call in (choice.delta.tool_calls or [])
    }
    assert indexes == {0, 1}


@pytest.mark.asyncio
async def test_a_truncated_response_is_reported_as_length():
    chunks = await _drain([
        _Event(type="RESPONSE_INCOMPLETE",
               response={"incomplete_details": {"reason": "max_output_tokens"}}),
    ])

    assert chunks[-1].choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_bookkeeping_events_yield_nothing():
    """Anything that carries no content must not reach the loop as an empty turn."""
    chunks = await _drain([
        _Event(type="RESPONSE_CREATED", response={}),
        _Event(type="RESPONSE_IN_PROGRESS", response={}),
        _Event(type="CONTENT_PART_ADDED", part={}),
        _Event(type="OUTPUT_TEXT_DONE", text="Hello"),
        _Event(type="CONTENT_PART_DONE", part={}),
        _Event(type="OUTPUT_ITEM_DONE", item={"type": "function_call"}),
    ])

    assert chunks == []


def test_the_request_is_rewritten_for_the_other_endpoint():
    from rune.agent.responses_bridge import build_responses_kwargs

    out = build_responses_kwargs({
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "max_tokens": 4096,
        "stream": True,
        "temperature": 0.7,
    })

    assert out["model"] == "gpt-6-astra"
    assert out["max_output_tokens"] == 4096
    assert "max_tokens" not in out
    assert "messages" not in out
    assert out["input"][0]["content"] == "hi"
    assert out["tools"][0]["name"] == "f"
    # temperature is not accepted by these models; it must not be forwarded.
    assert "temperature" not in out


# ---------------------------------------------------------------------------
# The rename that has to happen before anything else can be learned.
# ---------------------------------------------------------------------------

def test_the_max_tokens_rename_request_is_recognised():
    from rune.agent.model_traits import is_max_tokens_rename_error

    exc = Exception(
        "OpenAIException - Unsupported parameter: 'max_tokens' is not "
        "supported with this model. Use 'max_completion_tokens' instead."
    )

    assert is_max_tokens_rename_error(exc) is True
    assert is_max_tokens_rename_error(Exception("max_tokens is too large")) is False


class _BadRequest(Exception):
    pass


class _FakeClient:
    """Refuses in the order the live API does: rename first, endpoint second."""

    BadRequestError = _BadRequest

    def __init__(self, complaints):
        self._complaints = list(complaints)
        self.chat_calls: list[dict] = []
        self.responses_calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.chat_calls.append(dict(kwargs))
        if self._complaints:
            raise _BadRequest(self._complaints.pop(0))
        return "chat-stream"

    async def aresponses(self, **kwargs):
        self.responses_calls.append(dict(kwargs))
        return _EmptyStream()


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_a_rename_then_an_endpoint_refusal_both_get_handled():
    """The rename complaint arrives first and must not hide the endpoint one."""
    from rune.agent.litellm_adapter import _complete_dropping_rejected_params
    from rune.agent.model_traits import needs_responses_api

    client = _FakeClient([
        "Unsupported parameter: 'max_tokens' is not supported with this "
        "model. Use 'max_completion_tokens' instead.",
        "Function tools with reasoning_effort are not supported for "
        "gpt-6-astra in /v1/chat/completions. Use /v1/responses.",
    ])
    kwargs = {"model": "gpt-6-astra", "messages": [], "max_tokens": 100}

    await _complete_dropping_rejected_params(client, "gpt-6-astra", kwargs)

    assert len(client.chat_calls) == 2
    assert "max_completion_tokens" in client.chat_calls[1]
    assert "max_tokens" not in client.chat_calls[1]
    assert len(client.responses_calls) == 1
    assert needs_responses_api("gpt-6-astra") is True


@pytest.mark.asyncio
async def test_a_learned_model_skips_the_refusal_next_time():
    from rune.agent.litellm_adapter import _complete_dropping_rejected_params
    from rune.agent.model_traits import note_responses_only

    note_responses_only("gpt-6-astra")
    client = _FakeClient([])

    await _complete_dropping_rejected_params(
        client, "gpt-6-astra", {"model": "gpt-6-astra", "messages": []}
    )

    assert client.chat_calls == [], "spent a refusal it had already learned"
    assert len(client.responses_calls) == 1


@pytest.mark.asyncio
async def test_a_model_that_works_on_chat_is_left_alone():
    from rune.agent.litellm_adapter import _complete_dropping_rejected_params
    from rune.agent.model_traits import needs_responses_api

    client = _FakeClient([])

    result = await _complete_dropping_rejected_params(
        client, "gpt-5.4", {"model": "gpt-5.4", "messages": []}
    )

    assert result == "chat-stream"
    assert client.responses_calls == []
    assert needs_responses_api("gpt-5.4") is False


# ---------------------------------------------------------------------------
# Frontier defaults must be models that can actually run an agent step.
# ---------------------------------------------------------------------------

def test_every_default_tier_model_is_in_the_known_list():
    from rune.config.schema import ProviderModels
    from rune.llm.models import known_models

    known = {(provider, model_id) for provider, model_id in known_models()}
    providers = ProviderModels()

    for provider in ("openai", "anthropic"):
        tiers = getattr(providers, provider)
        for tier in ("best", "coding", "fast"):
            model_id = getattr(tiers, tier)
            assert (provider, model_id) in known, (
                f"{provider}.{tier} = {model_id} is not a listed model"
            )
