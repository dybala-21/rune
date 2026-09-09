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
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol"])
async def test_both_request_paths_preserve_the_completion_cap(monkeypatch, streaming, model):
    import litellm

    from rune.agent.litellm_adapter import StreamResult
    from rune.llm.client import LLMClient
    from rune.types import Provider
    from tests.unit.test_litellm_truncation_recovery import _astream, _delta_chunk

    calls = []
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}}}]

    async def api(**kwargs):
        calls.append(kwargs)
        if streaming:
            return _astream([_delta_chunk(content="done"), _delta_chunk(finish_reason="stop")])
        return {"answer": "done"}

    monkeypatch.setattr(litellm, "acompletion", api)
    if streaming:
        result = StreamResult(model=model, messages=[{"role": "user", "content": "hello"}],
                              tool_schemas=tools, tool_lookup={}, max_tokens=1024, temperature=0,
                              request_tokens_limit=10000, response_tokens_limit=1024)
        assert "done" in "".join([text async for text in result.stream_text()])
    else:
        assert await LLMClient().completion([{"role": "user", "content": "hello"}],
            model=model, provider=Provider.OPENAI, max_tokens=1024, tools=tools) == {"answer": "done"}
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 1024 and "max_tokens" not in calls[0]
    assert calls[0]["model"] == f"openai/responses/{model}"
    assert calls[0]["tools"] == tools
    assert calls[0]["store"] is False


@pytest.mark.parametrize('model, effort', [
    *[('gpt-6-astra', level) for level in ('low', 'medium', 'high', 'xhigh', 'max')],
    ('gpt-5.6-sol', 'max'), ('gpt-5.6-sol', 'none'),
    *[('anthropic/claude-opus-5', level) for level in ('low', 'medium', 'high', 'xhigh', 'max')],
    ('anthropic/claude-opus-4-6', 'max'), ('anthropic/claude-opus-4-5', 'high'),
    ('gemini/gemini-3-pro-preview', 'low'), ('gemini/gemini-3-pro-preview', 'high'),
    ('gemini/gemini-3.1-pro-preview', 'medium'), ('gemini/gemini-3-flash-preview', 'minimal'),
    ('gemini/gemini-2.5-pro', 'high'), ('gemini/gemini-2.5-flash', 'none'),
])
@pytest.mark.parametrize('streaming', [False, True])
async def test_reasoning_survives_litellm_wire_conversion(monkeypatch, model, effort, streaming):
    import json

    import httpx

    from rune.agent.litellm_adapter import _litellm

    calls = []

    async def send(self, request, **kwargs):
        calls.append((request.url.path, json.loads(request.content)))
        # Stop at the transport boundary: the real conversion ran, no API was contacted.
        return httpx.Response(400, request=request, json={
            'error': {'message': 'test boundary', 'type': 'invalid_request_error'},
        })

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    llm = _litellm()
    params = {'model': model, 'messages': [{'role': 'user', 'content': 'Say OK'}],
        'max_tokens': 2048, 'reasoning_effort': effort, 'temperature': 0,
        'stream': streaming, 'api_key': 'test-key', 'num_retries': 0,
        'tools': [{'type': 'function', 'function': {'name': 'read_file',
            'parameters': {'type': 'object', 'properties': {}}}}]}
    with pytest.raises(llm.BadRequestError, match='test boundary'):
        response = await compatible_completion(llm.acompletion, llm.BadRequestError, params)
        if streaming:
            async for _ in response:
                pass
    assert len(calls) == 1
    path, body = calls[0]
    assert 'reasoning_effort' not in body
    if model.startswith('gpt-'):
        assert path == '/v1/responses'
        assert body['model'] == model
        assert body['reasoning']['effort'] == effort
        assert body['max_output_tokens'] == 2048 and body['store'] is False
        assert body['tools'][0]['name'] == 'read_file'
        assert 'temperature' not in body
    elif model.startswith('anthropic/'):
        assert path == '/v1/messages'
        assert body['output_config']['effort'] == effort
        if '4-5' not in model:
            assert body['thinking'] == {'type': 'adaptive'}
            assert 'temperature' not in body
        else:
            assert 'thinking' not in body
        assert body['max_tokens'] == 2048
        assert body['tools'][0]['name'] == 'read_file'
    else:
        assert model.split('/', 1)[1] in path
        config = body['generationConfig']
        expected = {'thinkingBudget': 0 if effort == 'none' else 4096} if '2.5' in model else {'thinkingLevel': effort}
        assert config['thinkingConfig'] == expected
        assert config['max_output_tokens'] == 2048
        assert body['tools'][0]['function_declarations'][0]['name'] == 'read_file'


@pytest.mark.parametrize('model, effort', [
    ('gpt-6-astra', 'none'), ('gemini/gemini-3-pro-preview', 'medium'),
    ('anthropic/claude-opus-4-6', 'xhigh'), ('gemini/gemini-2.5-pro', 'none'),
])
async def test_unsupported_effort_is_rejected_before_transport(model, effort):
    from unittest.mock import AsyncMock

    api = AsyncMock()
    with pytest.raises(ValueError, match='Unsupported reasoning effort'):
        await compatible_completion(api, ValueError, {'model': model, 'reasoning_effort': effort})
    api.assert_not_called()


@pytest.mark.parametrize('model', ['gpt-6-astra', 'anthropic/claude-opus-5', 'gemini/gemini-3-pro-preview'])
async def test_default_does_not_add_reasoning_parameters(model):
    from unittest.mock import AsyncMock

    api = AsyncMock()
    await compatible_completion(api, ValueError, {'model': model, 'reasoning_effort': None})
    params = api.call_args.kwargs
    assert not {'reasoning_effort', 'thinking', 'thinkingConfig'} & params.keys()
    assert not {'reasoning', 'output_config'} & params.get('extra_body', {}).keys()


async def test_running_request_keeps_its_model_specific_preference(monkeypatch):
    import litellm

    from rune.agent.litellm_adapter import StreamResult
    from rune.config import get_config
    from tests.unit.test_litellm_truncation_recovery import _astream, _delta_chunk

    cfg = get_config().llm
    cfg.reasoning_efforts = {'openai/gpt-6-astra': 'high', 'anthropic/claude-opus-5': 'max'}
    result = StreamResult(model='gpt-6-astra', messages=[{'role': 'user', 'content': 'hello'}],
                          tool_schemas=[], tool_lookup={}, max_tokens=1024, temperature=0,
                          request_tokens_limit=10000, response_tokens_limit=1024)
    cfg.reasoning_efforts['openai/gpt-6-astra'] = 'low'
    calls = []

    async def api(**kwargs):
        calls.append(kwargs)
        return _astream([_delta_chunk(content='done'), _delta_chunk(finish_reason='stop')])

    monkeypatch.setattr(litellm, 'acompletion', api)
    assert 'done' in ''.join([text async for text in result.stream_text()])
    assert calls[0]['extra_body']['reasoning']['effort'] == 'high'
