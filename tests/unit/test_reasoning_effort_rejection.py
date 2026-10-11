"""Rejected reasoning settings must not silently change the user's request."""

import pytest

from rune.agent.model_traits import is_reasoning_effort_error
from rune.llm.request_params import compatible_completion
from rune.llm.target import ModelCompatibilityError


@pytest.mark.parametrize('message,expected', [
    ('Unsupported parameter: reasoning_effort is not supported', True),
    ('temperature does not support 0', False),
    ('maximum context is 8192 tokens', False),
    ('429 rate limit exceeded', False),
])
def test_reasoning_errors_are_distinct(message, expected):
    assert is_reasoning_effort_error(ValueError(message)) is expected


@pytest.mark.parametrize('model,field', [
    ('xai/grok-4.7', 'reasoning_effort'),
    ('gpt-6-astra', 'reasoning.effort'),
    ('anthropic/claude-opus-5', 'output_config.effort'),
    ('gemini/gemini-3.1-pro-preview', 'thinkingConfig'),
])
async def test_wire_rejection_preserves_effort_without_a_retry(model, field):
    calls = []
    async def api(**params):
        calls.append(params)
        raise ValueError(f'Unsupported parameter: {field}')
    with pytest.raises(ModelCompatibilityError, match='requested setting was preserved'):
        await compatible_completion(api, ValueError, {'model': model, 'reasoning_effort': 'low'})
    assert len(calls) == 1


async def test_endpoint_correction_preserves_reasoning():
    calls = []
    async def api(**params):
        calls.append(params)
        if '/responses/' not in params['model']:
            raise ValueError('Use /v1/responses instead of chat/completions')
        return 'ok'
    assert await compatible_completion(api, ValueError, {'model': 'openai/gpt-4o'}) == 'ok'
    assert len(calls) == 2


async def test_parameter_learning_is_scoped_to_the_endpoint():
    calls = []
    async def api(**params):
        calls.append(params)
        if params['api_base'] == 'http://localhost:8881' and 'temperature' in params:
            raise ValueError('temperature is not supported')
        return 'ok'
    params = {'model': 'openai/contract-fixture', 'temperature': .5, 'api_base': 'http://localhost:8881'}
    await compatible_completion(api, ValueError, params)
    await compatible_completion(api, ValueError, params)
    await compatible_completion(api, ValueError, {**params, 'api_base': 'http://localhost:8882'})
    assert len(calls) == 4
    assert 'temperature' not in calls[2]
    assert calls[3]['temperature'] == .5
