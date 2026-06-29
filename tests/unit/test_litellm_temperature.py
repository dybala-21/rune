"""Temperature handling for models that only accept temperature=1.

drop_params lets litellm strip it for the models it knows (gpt-5, o-series); the
retry path catches the ones it doesn't (gpt-5.5, claude-opus-4-8 reject it but
litellm reports it supported). See litellm_adapter.
"""

from __future__ import annotations

import litellm

import rune.agent.litellm_adapter  # noqa: F401  (enables litellm.drop_params)
from rune.agent.litellm_adapter import _TEMPERATURE_REJECTED, _is_temperature_error


def test_import_enables_drop_params():
    assert litellm.drop_params is True


def _temperature_kept(model: str, provider: str) -> bool:
    opt = litellm.get_optional_params(
        model=model, custom_llm_provider=provider, temperature=0.0, drop_params=True)
    return "temperature" in opt and opt["temperature"] is not None


def test_litellm_drops_temperature_for_restricted_models():
    assert not _temperature_kept("gpt-5", "openai")
    assert not _temperature_kept("gpt-5.4-pro", "openai")
    assert not _temperature_kept("o3-mini", "openai")


def test_detects_temperature_errors():
    # real messages from the live APIs
    assert _is_temperature_error(Exception(
        "OpenAIException - Unsupported value: 'temperature' does not support 0 "
        "with this model. Only the default (1) value is supported."))
    assert _is_temperature_error(Exception(
        "AnthropicException - `temperature` is deprecated for this model."))


def test_ignores_unrelated_errors():
    assert not _is_temperature_error(Exception("rate_limit_exceeded"))
    assert not _is_temperature_error(Exception("invalid x-api-key"))
    assert not _is_temperature_error(Exception("context_length_exceeded"))


def test_learned_set_membership():
    _TEMPERATURE_REJECTED.discard("probe-model")
    assert "probe-model" not in _TEMPERATURE_REJECTED
    _TEMPERATURE_REJECTED.add("probe-model")
    assert "probe-model" in _TEMPERATURE_REJECTED
    _TEMPERATURE_REJECTED.discard("probe-model")
