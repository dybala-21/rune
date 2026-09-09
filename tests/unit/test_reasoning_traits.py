"""Reasoning choices must describe an actual, model-specific control."""

import pytest

from rune.agent.model_traits import (
    effective_reasoning_effort,
    reasoning_efforts,
    supports_reasoning_effort,
)
from rune.llm.reasoning import configured_reasoning_effort, reasoning_control


@pytest.fixture(autouse=True)
def clear_metadata_cache():
    for fn in (supports_reasoning_effort, reasoning_efforts, reasoning_control):
        fn.cache_clear()
    yield
    for fn in (supports_reasoning_effort, reasoning_efforts, reasoning_control):
        fn.cache_clear()


@pytest.mark.parametrize('model, choices', [
    ('gpt-6-astra', ('low', 'medium', 'high', 'xhigh', 'max')),
    ('openai/responses/gpt-6-astra', ('low', 'medium', 'high', 'xhigh', 'max')),
    ('gpt-5.6-sol', ('none', 'low', 'medium', 'high', 'xhigh', 'max')),
    ('gpt-5.6', ('none', 'low', 'medium', 'high', 'xhigh', 'max')),
    ('anthropic/claude-opus-5', ('low', 'medium', 'high', 'xhigh', 'max')),
    ('anthropic/claude-opus-4-6', ('low', 'medium', 'high', 'max')),
    ('anthropic/claude-opus-4-5-20251101', ('low', 'medium', 'high')),
    ('gemini/gemini-3-pro-preview', ('low', 'high')),
    ('vertex_ai/gemini-3.1-pro-preview', ('low', 'medium', 'high')),
    ('gemini/gemini-3-flash-preview', ('minimal', 'low', 'medium', 'high')),
    ('gemini/gemini-2.5-pro', ('low', 'medium', 'high')),
    ('gemini/gemini-2.5-flash', ('none', 'low', 'medium', 'high')),
])
def test_documented_controls_work_offline(monkeypatch, model, choices):
    import litellm

    monkeypatch.setattr(litellm, 'get_model_info', lambda *a, **kw: {})
    assert reasoning_efforts(model) == choices
    assert effective_reasoning_effort(model, choices[-1]) == choices[-1]
    assert effective_reasoning_effort(model, 'unsupported') is None


@pytest.mark.parametrize('model', [
    'private/gpt-6-astra', 'ollama/gpt-6-astra', 'gpt-6-astra-mini',
    'anthropic/claude-opus-5-mini', 'gemini/gemini-3-pro-image-preview', 'private/unknown',
])
def test_unknown_models_do_not_inherit_controls(monkeypatch, model):
    import litellm

    monkeypatch.setattr(litellm, 'supports_reasoning', lambda **kw: True)
    monkeypatch.setattr(litellm, 'get_supported_openai_params', lambda **kw: ['reasoning_effort'])
    monkeypatch.setattr(litellm, 'get_model_info', lambda *a, **kw: {})
    assert not supports_reasoning_effort(model)


@pytest.mark.parametrize('params', [[], ['reasoning_effort']])
def test_metadata_requires_explicit_levels_and_a_supported_parameter(monkeypatch, params):
    import litellm

    calls = []
    monkeypatch.setattr(litellm, 'get_supported_openai_params', lambda **kw: params)

    def lookup(model):
        calls.append(model)
        return {'supports_none_reasoning_effort': True, 'supports_low_reasoning_effort': False,
                'supports_xhigh_reasoning_effort': True, 'supports_max_reasoning_effort': None}

    monkeypatch.setattr(litellm, 'get_model_info', lookup)
    for _ in range(2):
        assert reasoning_efforts('private/reasoner') == (('none', 'xhigh') if params else ())
    assert calls == ['private/reasoner']


def test_unavailable_metadata_disables_the_control(monkeypatch):
    import litellm

    def unavailable(**kwargs):
        raise ValueError('No metadata')

    monkeypatch.setattr(litellm, 'get_supported_openai_params', unavailable)
    assert reasoning_efforts('private/unknown') == ()


@pytest.fixture(params=['reasoningEffort', 'reasoning_effort'])
def preferences(tmp_path, monkeypatch, request):
    import json

    from rune.config import load_config

    monkeypatch.setenv('RUNE_HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'config.yaml'
    path.write_text(json.dumps({'llm': {
        'activeProvider': 'openai', 'activeModel': 'gpt-6-astra', request.param: 'high',
    }}))
    load_config(force=True)
    return path


async def test_preferences_survive_model_switches_and_restarts(preferences):
    from rune.api.handlers.config import get_config_endpoint, set_reasoning_effort
    from rune.config import get_config, load_config
    from rune.llm.model_selection import ActiveModelSelection, persist_active_model_selection
    from rune.types import Provider

    assert configured_reasoning_effort('gpt-6-astra') == 'high'
    assert configured_reasoning_effort('gpt-5.6-sol') is None
    persist_active_model_selection(ActiveModelSelection(Provider.OPENAI, 'gpt-5.6-sol'))
    load_config(force=True)
    assert configured_reasoning_effort('gpt-6-astra') == 'high'
    assert (await get_config_endpoint()).reasoning_effort is None
    await set_reasoning_effort('none', provider='openai', model='gpt-5.6-sol')
    assert configured_reasoning_effort('openai/responses/gpt-5.6-sol') == 'none'
    await set_reasoning_effort('', provider='openai', model='gpt-5.6-sol')
    load_config(force=True)
    assert get_config().llm.reasoning_efforts['openai/gpt-5.6-sol'] is None
    persist_active_model_selection(ActiveModelSelection(Provider.OPENAI, 'gpt-6-astra'))
    assert (await get_config_endpoint()).reasoning_effort == 'high'
    await set_reasoning_effort('max', provider='openai', model='gpt-6-astra')
    load_config(force=True)
    assert configured_reasoning_effort('gpt-6-astra') == 'max'


@pytest.mark.parametrize('effort, model', [('none', 'gpt-6-astra'), ('max', 'gpt-5.6-sol'), (None, 'gpt-6-astra')])
async def test_invalid_or_stale_selection_does_not_write(preferences, effort, model):
    from rune.api.handlers.config import set_reasoning_effort

    before = preferences.read_bytes()
    with pytest.raises(ValueError):
        await set_reasoning_effort(effort, provider='openai', model=model)
    assert preferences.read_bytes() == before
    assert configured_reasoning_effort('gpt-6-astra') == 'high'


async def test_model_changed_during_lookup_does_not_receive_old_preference(preferences, monkeypatch):
    from rune.api.handlers.config import set_reasoning_effort
    from rune.config import get_config

    def changed(model):
        get_config().llm.active_model = 'gpt-5.6-sol'
        return ('max',)

    monkeypatch.setattr('rune.agent.model_traits.reasoning_efforts', changed)
    before = preferences.read_bytes()
    with pytest.raises(ValueError, match='model changed'):
        await set_reasoning_effort('max', provider='openai', model='gpt-6-astra')
    assert preferences.read_bytes() == before
    assert configured_reasoning_effort('gpt-5.6-sol') is None


async def test_failed_save_does_not_change_live_settings(preferences, monkeypatch):
    from rune.api.handlers.config import set_reasoning_effort
    from rune.llm.model_selection import (
        ActiveModelSelection,
        get_effective_model_selection,
        persist_active_model_selection,
    )
    from rune.types import Provider

    monkeypatch.setattr('rune.config.save_config_values', lambda updates: None)
    with pytest.raises(OSError, match='save reasoning'):
        await set_reasoning_effort('max', provider='openai', model='gpt-6-astra')
    with pytest.raises(OSError, match='save model'):
        persist_active_model_selection(ActiveModelSelection(Provider.ANTHROPIC, 'claude-opus-5'))
    assert configured_reasoning_effort('gpt-6-astra') == 'high'
    assert get_effective_model_selection().model == 'gpt-6-astra'


def test_explicit_default_wins_over_legacy_global_setting():
    from rune.config.schema import LLMConfig

    cfg = LLMConfig(activeProvider='openai', activeModel='gpt-6-astra', reasoningEffort='high',
                    reasoningEfforts={'openai/gpt-6-astra': None})
    assert cfg.reasoning_efforts == {'openai/gpt-6-astra': None}
