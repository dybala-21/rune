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
    mt._COMPLETION_TOKENS_REQUIRED.clear()
    yield
    mt._RESPONSES_ONLY.clear()
    mt._COMPLETION_TOKENS_REQUIRED.clear()


async def test_token_rename_and_endpoint_fallback_are_remembered():
    from rune.llm.request_params import compatible_completion

    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise ValueError("Unsupported max_tokens; use max_completion_tokens instead")
        if "/responses/" not in kwargs["model"]:
            raise ValueError(NOT_ON_THIS_ENDPOINT)
        return "answer"

    params = {"model": "openai/future-model", "max_tokens": 100, "messages": []}
    assert await compatible_completion(api, ValueError, params) == "answer"
    assert len(calls) == 3
    assert calls[-1]["model"] == "openai/responses/future-model"
    assert calls[-1]["max_completion_tokens"] == 100
    assert calls[-1]["store"] is False
    assert await compatible_completion(api, ValueError, params) == "answer"
    assert len(calls) == 4
    assert params["model"] == "openai/future-model"


async def test_endpoint_refusal_preserves_effort_before_parameter_fallback(monkeypatch):
    from rune.agent import model_traits as mt
    from rune.llm.reasoning import ReasoningControl
    from rune.llm.request_params import compatible_completion

    monkeypatch.setattr("rune.llm.reasoning.reasoning_control", lambda model: ReasoningControl(("high",)))
    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        if "/responses/" not in kwargs["model"]:
            raise ValueError(CHAT_TOOLS_REFUSED)
        return "answer"

    await compatible_completion(api, ValueError, {"model": "openai/future-model", "reasoning_effort": "high"})
    assert len(calls) == 2
    assert calls[-1]["extra_body"]["reasoning"]["effort"] == "high"
    assert not mt.reasoning_effort_rejected("openai/future-model")


@pytest.mark.parametrize("model", ["openai/responses/future-model", "anthropic/claude-sonnet-5"])
async def test_endpoint_refusal_cannot_retry_forever_or_change_provider(model):
    from rune.llm.request_params import compatible_completion

    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        raise ValueError(NOT_ON_THIS_ENDPOINT)

    with pytest.raises(ValueError, match="endpoint"):
        await compatible_completion(api, ValueError, {"model": model})
    assert len(calls) == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_responses_wire_keeps_images_tool_history_and_tool_choice(monkeypatch, streaming):
    import json

    import httpx

    from rune.agent.litellm_adapter import _litellm
    from rune.llm.request_params import compatible_completion

    calls = []

    async def send(self, request, **kwargs):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(400, request=request, json={"error": {"message": "test boundary"}})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    llm = _litellm()
    photo = "data:image/png;base64,iVBORw0KGgo="
    params = {
        "model": "gpt-6-astra", "reasoning_effort": "max", "stream": streaming,
        "api_key": "test-key", "num_retries": 0,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Read this chart"},
                {"type": "image_url", "image_url": {"url": photo}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sales: 391"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file",
            "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
    }
    with pytest.raises(llm.BadRequestError, match="test boundary"):
        result = await compatible_completion(llm.acompletion, llm.BadRequestError, params)
        if streaming:
            async for _ in result:
                pass
    assert len(calls) == 1
    path, body = calls[0]
    assert path == "/v1/responses"
    assert body["reasoning"] == {"effort": "max"}
    assert body["tool_choice"] == {"type": "function", "name": "read_file"}
    assert any(part.get("image_url") == photo for item in body["input"] for part in item.get("content", []))
    assert any(item.get("call_id") == "call_1" and item.get("type") == "function_call" for item in body["input"])
    output = next(item["output"] for item in body["input"] if item.get("type") == "function_call_output")
    assert "sales: 391" in str(output)


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
