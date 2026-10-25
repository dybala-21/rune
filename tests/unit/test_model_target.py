"""Main and auxiliary calls must keep their original provider and endpoint."""

from unittest.mock import AsyncMock

import pytest

from rune.agent.litellm_adapter import LiteLLMAgent, _resolve_litellm_model
from rune.llm.context import model_scope
from rune.llm.target import ModelCompatibilityError, resolve_target


@pytest.mark.parametrize("model,provider", [("grok-4.7", "xai"), ("gpt-6-astra", "openai"),
                                          ("claude-opus-5", "anthropic"), ("qwen3:8b", "ollama")])
def test_model_preflight_accepts_priced_cloud_and_local_models(model, provider):
    from rune.llm.target import preflight_model

    assert preflight_model(model, provider, cost_limit=.30).provider == provider


def test_model_preflight_blocks_unknown_pricing_only_when_a_cost_limit_is_set():
    from rune.llm.target import preflight_model

    assert preflight_model("unpriced-fixture", "openai").model == "unpriced-fixture"
    with pytest.raises(ModelCompatibilityError, match="Pricing is unavailable"):
        preflight_model("unpriced-fixture", "openai", cost_limit=.30)


@pytest.mark.parametrize("case", ["region", "pricing"])
async def test_run_preflight_happens_before_any_routing_call(monkeypatch, tmp_path, case):
    from rune.agent.loop import NativeAgentLoop
    from rune.config import get_config
    from rune.types import AgentConfig

    cfg = get_config()
    if case == "region":
        credentials = tmp_path / "credentials.json"
        credentials.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
        monkeypatch.setenv("VERTEX_LOCATION", "us-central1")
        cfg.llm.active_provider, cfg.llm.active_model = "gemini", "gemini-3.1-pro-preview"
        error = "not available in us-central1"
    else:
        cfg.llm.active_provider, cfg.llm.active_model = "openai", "unpriced-fixture"
        error = "Pricing is unavailable"
    classify = AsyncMock()
    monkeypatch.setattr("rune.agent.loop.classify_goal", classify)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    loop = NativeAgentLoop(AgentConfig(cost_budget_usd=.30))
    trace = await loop.run("Research recent notices")
    assert error in trace.reason
    classify.assert_not_awaited()
    assert not loop._running


async def test_local_file_roles_keep_local_transport_and_identity(monkeypatch, tmp_path):
    import json

    import httpx

    from rune.llm.pricing import estimate_request_cost

    calls = []
    async def send(self, request, **kwargs):
        calls.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, request=request, json={
            'id': 'local-contract', 'object': 'chat.completion', 'created': 0, 'model': 'qwen3:8b',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': '{"notes.txt":"input"}',
            }}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8},
        })
    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    agent = LiteLLMAgent(model='ollama/qwen3:8b', tools=[])
    async with agent.run_stream('Read notes.txt', workspace_root=str(tmp_path)) as stream:
        await stream._classify_artifact_roles()
        assert stream._ledger().roles == {'notes.txt': 'input'}
    assert len(calls) == 1
    assert calls[0][0] == 'http://localhost:11434/v1/chat/completions'
    assert calls[0][1]['model'] == 'qwen3:8b'
    assert '_rune_source_model' not in calls[0][1]
    assert estimate_request_cost(agent._model, {'input_tokens': 5, 'output_tokens': 3}, agent._provider_extra) == 0


@pytest.mark.parametrize('model', ['gemini-3.1-pro-preview', 'gemini/gemini-3.1-pro-preview', 'vertex_ai/gemini-3.1-pro-preview'])
async def test_vertex_resolution_is_shared_and_never_double_prefixed(monkeypatch, tmp_path, model):
    from rune.agent.litellm_adapter import litellm
    from rune.llm.client import LLMClient
    from rune.types import Provider
    credentials = tmp_path / 'credentials.json'
    credentials.write_text('{}')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(credentials))
    monkeypatch.setenv('VERTEX_LOCATION', 'auto')
    complete = AsyncMock(return_value={})
    monkeypatch.setattr(litellm, 'acompletion', complete)
    await LLMClient().completion([], model=model, provider=Provider.GEMINI)
    sent = complete.call_args.kwargs
    assert sent['model'] == 'vertex_ai/gemini-3.1-pro-preview'
    assert sent['vertex_location'] == 'global'
    monkeypatch.setenv('VERTEX_LOCATION', 'us-central1')
    with pytest.raises(ModelCompatibilityError, match='not available'):
        await LLMClient().completion([], model=model, provider=Provider.GEMINI)
    assert complete.await_count == 1
    with pytest.raises(ModelCompatibilityError):
        _resolve_litellm_model('gemini/gemini-3.1-pro-preview')


def test_run_settings_and_region_do_not_change_mid_run(monkeypatch):
    from rune.config import get_config
    from rune.llm.model_selection import get_effective_model_selection
    cfg = get_config().llm
    monkeypatch.setattr(cfg, 'active_provider', 'xai')
    monkeypatch.setattr(cfg, 'active_model', 'grok-4.7')
    monkeypatch.setenv('VERTEX_LOCATION', 'global')
    with model_scope():
        cfg.active_provider, cfg.active_model = 'ollama', 'qwen3:8b'
        monkeypatch.setenv('VERTEX_LOCATION', 'us-central1')
        with model_scope():
            assert get_effective_model_selection().provider == 'xai'
            assert resolve_target('vertex_ai/gemini-3.1-pro-preview').kwargs()['vertex_location'] == 'global'
    assert get_effective_model_selection().provider == 'ollama'


def test_new_xai_model_uses_supported_sdk_metadata(monkeypatch):
    import litellm

    from rune.llm.reasoning import _reasoning_control, reasoning_control
    _reasoning_control.cache_clear()
    monkeypatch.setattr(litellm, 'get_supported_openai_params', lambda **_: ['reasoning_effort'])
    monkeypatch.setattr(litellm, 'get_model_info', lambda *_: {'supports_low_reasoning_effort': True})
    try:
        control = reasoning_control('xai/grok-contract-fixture')
        assert control.efforts == ('low',) and control.wire == 'xai'
    finally:
        _reasoning_control.cache_clear()


async def test_initial_routing_reuses_file_roles_without_an_extra_request(monkeypatch):
    from rune.agent.goal_classifier import classify_tier2
    from tests.unit.test_goal_classifier import response, verdict
    client = AsyncMock()
    client.completion.return_value = response(verdict(file_roles=[
        {'name': 'notes.txt', 'role': 'input'}, {'name': 'report.txt', 'role': 'output'},
    ]))
    monkeypatch.setattr('rune.llm.client.get_llm_client', lambda: client)
    request = 'Read notes.txt and create report.txt.'
    result = await classify_tier2(request)
    assert result.available
    assert result.artifact_roles.matching_roles(request) == {'notes.txt': 'input', 'report.txt': 'output'}
    assert client.completion.await_count == 1
    contract = client.completion.call_args.kwargs['response_format']
    assert 'file_roles' in contract['json_schema']['schema']['required']


def test_scoped_credentials_and_local_targets_stay_separate(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-initial')
    with model_scope():
        monkeypatch.setenv('OPENAI_API_KEY', 'test-replacement')
        target = resolve_target('openai/example')
        assert target.kwargs()['api_key'] == 'test-initial'
        assert 'test-initial' not in repr(target)
        assert resolve_target('ollama/example').kwargs()['api_key'] == 'ollama'
    assert resolve_target('openai/example').kwargs()['api_key'] == 'test-replacement'
    with pytest.raises(ModelCompatibilityError, match='belongs to'):
        resolve_target('openai/example', 'ollama')


def test_local_failure_and_incompatible_settings_do_not_switch_provider(monkeypatch):
    from rune.agent.failover import build_profiles_from_config, classify_error, determine_strategy
    from rune.config import get_config

    monkeypatch.setattr(get_config().llm, 'active_provider', 'ollama')
    monkeypatch.setattr(get_config().llm, 'active_model', 'qwen3:8b')
    profiles = build_profiles_from_config()
    assert [(profile.provider, profile.model) for profile in profiles] == [('ollama', 'qwen3:8b')]
    assert determine_strategy('auth', profiles[0], 0, profiles).action == 'abort'
    reason = classify_error(ModelCompatibilityError('Unsupported region'))
    assert reason == 'invalid_request'
    assert determine_strategy(reason, profiles[0], 3, profiles).action == 'abort'


def test_new_vertex_models_use_the_configured_endpoint_policy(monkeypatch):
    monkeypatch.setenv('VERTEX_LOCATION', 'auto')
    assert resolve_target('vertex_ai/gemini-future-fixture').kwargs()['vertex_location'] == 'global'
    monkeypatch.setenv('VERTEX_LOCATION', 'europe-west4')
    assert resolve_target('vertex_ai/gemini-future-fixture').kwargs()['vertex_location'] == 'europe-west4'
