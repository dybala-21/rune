"""Grok selection, discovery, and request compatibility."""

from unittest.mock import AsyncMock

import httpx
import pytest

from rune.llm import xai


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    from rune.config.loader import reset_config

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    reset_config()
    xai.invalidate_cache()
    yield
    xai.invalidate_cache()
    reset_config()


async def test_discovery_uses_account_models_and_caches_by_credential(monkeypatch):
    from rune.llm.models import selectable_models

    calls = []

    def reply(request):
        calls.append(request.headers["authorization"])
        assert str(request.url) == "https://api.x.ai/v1/language-models"
        return httpx.Response(200, json={"models": [
            {"id": "grok-4.3", "created": 10, "output_modalities": ["text"]},
            {"id": "grok-new", "created": 20, "output_modalities": ["text"]},
            {"id": "grok-new", "created": 20, "output_modalities": ["text"]},
            {"id": "grok-4.20-multi-agent", "created": 30, "output_modalities": ["text"]},
            {"id": "grok-imagine", "created": 30, "output_modalities": ["image"]},
        ]})

    client = httpx.AsyncClient
    monkeypatch.setattr(xai.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(reply), **kwargs))
    monkeypatch.setenv("XAI_API_KEY", "test-key-one")
    for _ in range(2):
        models = await selectable_models()
        assert [name for provider, name in models if provider == "xai"] == ["grok-new", "grok-4.3"]
        assert ("openai", "gpt-6-astra") in models
    assert calls == ["Bearer test-key-one"]
    monkeypatch.setenv("XAI_API_KEY", "test-key-two")
    await selectable_models()
    assert calls == ["Bearer test-key-one", "Bearer test-key-two"]


@pytest.mark.parametrize("status,payload,expected", [
    (503, {}, list(xai.MODELS)),
    (200, {"models": "invalid"}, list(xai.MODELS)),
    (200, {"models": []}, []),
])
async def test_catalog_failure_keeps_fallback_but_empty_access_does_not(monkeypatch, status, payload, expected):
    from rune.llm.models import selectable_models

    client = httpx.AsyncClient
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(xai.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload)), **kwargs))
    assert [name for provider, name in await selectable_models() if provider == "xai"] == expected


async def test_selection_survives_reload_and_auxiliary_calls_stay_on_xai(monkeypatch):
    import rune.agent.litellm_adapter as adapter
    from rune.cli.main import _ensure_llm_key
    from rune.config import get_config, load_config
    from rune.llm.client import LLMClient
    from rune.llm.model_selection import (
        ActiveModelSelection,
        get_active_model_selection,
        persist_active_model_selection,
    )
    from rune.types import ModelTier, Provider

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    selection = ActiveModelSelection(Provider.XAI, "grok-4.6")
    persist_active_model_selection(selection)
    load_config(force=True)
    assert get_active_model_selection() == selection
    assert _ensure_llm_key()
    client = LLMClient()
    assert client._effective_provider(None) == Provider.XAI
    assert client.resolve_model(ModelTier.FAST) == get_config().llm.models.xai.fast
    assert adapter._resolve_litellm_model("xai:grok-4.6") == ("xai/grok-4.6", {})
    assert adapter._resolve_litellm_model("xai/grok-4.6") == ("xai/grok-4.6", {})
    completion = AsyncMock(return_value={"choices": []})
    monkeypatch.setattr(adapter.litellm, "acompletion", completion)
    await client.completion([{"role": "user", "content": "classify"}], tier=ModelTier.FAST)
    assert completion.call_args.kwargs["model"] == "xai/grok-4.3"


@pytest.mark.parametrize("model,efforts", [
    ("xai/grok-4.6", ("low", "medium", "high", "xhigh")),
    ("grok-4.5", ("low", "medium", "high", "xhigh")),
    ("xai/grok-4.3-latest", ("none", "low", "medium", "high", "xhigh")),
    ("xai/grok-code-fast-1", ()),
    ("xai/grok-4.20-0309-non-reasoning", ()),
])
def test_grok_capabilities_are_model_specific(model, efforts):
    from rune.agent.model_traits import supports_vision
    from rune.llm.reasoning import apply_reasoning_control, reasoning_control
    from rune.llm.structured import supported_format

    assert supports_vision(model)
    assert reasoning_control(model).efforts == efforts
    schema = {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}}
    assert supported_format(model, schema) == schema
    for effort in efforts:
        params = {"model": model, "reasoning_effort": effort, "extra_body": {"existing": True}}
        apply_reasoning_control(params)
        assert params["extra_body"] == {"existing": True, "reasoning_effort": effort}
        assert "reasoning_effort" not in params
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        apply_reasoning_control({"model": model, "reasoning_effort": "max"})


def test_private_models_do_not_inherit_xai_capabilities():
    from rune.llm.reasoning import reasoning_control

    assert xai.model_name("private/grok-4.6") is None
    assert xai.model_name("xai/grok-4.6-mini") is None
    assert reasoning_control("xai/grok-unknown").efforts == ()


def test_xai_key_is_visible_as_a_masked_llm_setting():
    from rune.api.handlers.env import _categorize_key, _is_rune_key, _mask_value

    assert _is_rune_key("XAI_API_KEY")
    assert _categorize_key("XAI_API_KEY") == "llm"
    assert "test-secret" not in _mask_value("XAI_API_KEY", "test-secret")
