"""Keep the requested model usable when its API rejects legacy parameters."""

import pytest

from rune.agent.failover import LLMProfile, classify_error, determine_strategy
from rune.llm.request_params import compatible_completion

TOKEN_ERROR = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."


@pytest.fixture(autouse=True)
def overlays(monkeypatch):
    monkeypatch.setattr("rune.agent.model_traits._COMPLETION_TOKENS_REQUIRED", set())
    monkeypatch.setattr("rune.agent.model_traits._TEMPERATURE_REJECTED", set())


async def test_recovers_both_rejections_and_remembers_the_exact_model():
    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise ValueError(TOKEN_ERROR)
        if "temperature" in kwargs:
            raise ValueError("Unsupported value: temperature does not support 0")
        return "answer"

    params = {"model": "openai/new-reasoner", "max_tokens": 1024, "temperature": 0}
    assert await compatible_completion(api, ValueError, params) == "answer"
    assert len(calls) == 3
    assert calls[-1] == {"model": params["model"], "max_completion_tokens": 1024}
    assert "max_tokens" in params  # Caller configuration was not changed.
    assert await compatible_completion(api, ValueError, params) == "answer"
    assert len(calls) == 4


@pytest.mark.parametrize("message", ["maximum context length exceeded", "rate limit", "invalid API key"])
async def test_unrelated_errors_are_not_retried(message):
    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        raise ValueError(message)

    with pytest.raises(ValueError, match=message):
        await compatible_completion(api, ValueError, {"model": "other", "max_tokens": 10})
    assert len(calls) == 1


async def test_repeated_parameter_rejection_is_bounded():
    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        raise ValueError(TOKEN_ERROR)

    with pytest.raises(ValueError, match="Unsupported"):
        await compatible_completion(api, ValueError, {"model": "other", "max_tokens": 10})
    assert len(calls) == 2


def test_bad_parameter_does_not_trigger_context_compaction_or_model_switch():
    reason = classify_error(TOKEN_ERROR)
    profile = LLMProfile(name="main", model="gpt-6-astra", provider="openai")
    assert reason == "invalid_request"
    assert determine_strategy(reason, profile, 3, [profile]).action == "abort"
    assert classify_error("max_tokens exceeds maximum context length") == "context_overflow"


@pytest.mark.parametrize("streaming", [False, True])
async def test_both_request_paths_preserve_the_completion_cap(monkeypatch, streaming):
    import litellm

    from rune.agent.litellm_adapter import StreamResult
    from rune.llm.client import LLMClient
    from rune.types import Provider
    from tests.unit.test_litellm_truncation_recovery import _astream, _delta_chunk

    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        if streaming:
            return _astream([_delta_chunk(content="done"), _delta_chunk(finish_reason="stop")])
        return {"answer": "done"}

    monkeypatch.setattr(litellm, "acompletion", api)
    if streaming:
        result = StreamResult(model="gpt-6-astra", messages=[{"role": "user", "content": "hello"}],
                              tool_schemas=[], tool_lookup={}, max_tokens=1024, temperature=0,
                              request_tokens_limit=10000, response_tokens_limit=1024)
        assert "done" in "".join([text async for text in result.stream_text()])
    else:
        assert await LLMClient().completion([{"role": "user", "content": "hello"}],
            model="gpt-6-astra", provider=Provider.OPENAI, max_tokens=1024) == {"answer": "done"}
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 1024 and "max_tokens" not in calls[0]
    assert calls[0]["model"] == "openai/responses/gpt-6-astra"
    assert calls[0]["store"] is False
