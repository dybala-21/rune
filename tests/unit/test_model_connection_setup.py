"""Offline onboarding metadata must follow the selected transport without probing it."""

import json

import pytest

from rune.api.model_connection import model_connection
from rune.config.schema import CompatibleEndpoint, RuneConfig
from rune.llm.models import PROVIDER_ENV_KEYS


@pytest.fixture(autouse=True)
def isolated_engine(monkeypatch, tmp_path):
    from rune.config import loader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    keys = {key for names in PROVIDER_ENV_KEYS.values() for key in names} | {
        "GOOGLE_APPLICATION_CREDENTIALS", "VERTEX_LOCATION", "VERTEXAI_LOCATION",
        "VERTEX_PROJECT", "VERTEXAI_PROJECT", "AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT",
        "AZURE_AD_TOKEN", "RUNE_LLM_GATEWAY_URL", "RUNE_LLM_GATEWAY_KEY", "RUNE_CLOUD_WORKER",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("rune.cloud.boundary._HOSTED", False)
    cfg = RuneConfig()
    monkeypatch.setattr(loader, "_config", cfg)

    def no_probe(*args, **kwargs):
        raise AssertionError("setup metadata must not probe or construct a provider client")

    monkeypatch.setattr("socket.getaddrinfo", no_probe)
    monkeypatch.setattr("httpx.Client", no_probe)
    monkeypatch.setattr("httpx.AsyncClient", no_probe)
    monkeypatch.setattr("rune.llm.client.get_llm_client", no_probe)
    monkeypatch.setattr("rune.llm.target.resolve_target", no_probe)
    return cfg


@pytest.mark.parametrize(("provider", "model", "key"), [
    ("openai", "gpt-6-sol", "OPENAI_API_KEY"),
    ("anthropic", "claude-sonnet-5", "ANTHROPIC_API_KEY"),
    ("xai", "grok-4.6", "XAI_API_KEY"),
    ("gemini", "gemini-2.5-flash", "GEMINI_API_KEY"),
])
def test_only_selected_provider_credentials_count(provider, model, key, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "unrelated-secret")
    missing = model_connection(provider, model)
    assert missing.state == "needs_setup"
    assert missing.action.key == key
    assert missing.provider == provider and missing.model == model
    monkeypatch.setenv(key, "private-provider-secret")
    ready = model_connection(provider, model)
    assert ready.state == "configured"
    assert "not been verified" in ready.reason
    assert "private-provider-secret" not in ready.model_dump_json()
    monkeypatch.setenv(key, "   ")
    assert model_connection(provider, model).state == "needs_setup"


def test_cached_config_keys_do_not_claim_runtime_auth(isolated_engine):
    isolated_engine.openai_api_key = "config-only-secret"
    isolated_engine.anthropic_api_key = "config-only-secret"
    assert model_connection("openai", "gpt-6-sol").state == "needs_setup"
    assert model_connection("anthropic", "claude-sonnet-5").state == "needs_setup"


def test_gemini_api_alias_and_vertex_paths_are_distinguished(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "api-key-secret")
    assert model_connection("gemini", "gemini-2.5-flash").state == "configured"
    credentials = tmp_path / "service-account.json"
    credentials.write_text("not parsed by setup metadata")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    vertex = model_connection("gemini", "gemini-2.5-flash")
    assert vertex.state == "unknown" and "Vertex" in vertex.reason
    assert str(credentials) not in vertex.model_dump_json()
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")
    unsupported = model_connection("gemini", "gemini-3.1-pro-preview")
    assert unsupported.state == "needs_setup" and unsupported.action.key == "VERTEX_LOCATION"
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
    assert model_connection("gemini", "gemini-2.5-flash").state == "needs_setup"


def test_explicit_vertex_adc_cannot_be_verified_offline():
    result = model_connection("gemini", "vertex_ai/gemini-2.5-flash")
    assert result.state == "unknown" and "cloud credentials" in result.reason


def test_azure_requires_endpoint_credential_and_identifies_deployment(isolated_engine, monkeypatch):
    assert model_connection("azure", "daily").state == "needs_setup"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert model_connection("azure", "daily").state == "needs_setup"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    assert model_connection("azure", "daily").state == "unknown"
    isolated_engine.llm.azure_deployments = {"daily": "gpt-6-sol"}
    state = model_connection("azure", "daily")
    assert state.state == "configured"
    assert "azure-secret" not in state.model_dump_json()
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    monkeypatch.setenv("AZURE_AD_TOKEN", "azure-token-secret")
    assert model_connection("azure", "daily").state == "unknown"


def test_local_endpoints_do_not_require_keys_or_invent_model_availability(isolated_engine):
    assert model_connection("ollama", "qwen3-coder:30b").state == "unknown"
    assert model_connection("openai_compatible", "studio/local-model").state == "needs_setup"
    isolated_engine.llm.endpoints = {"studio": CompatibleEndpoint(baseUrl="http://localhost:1234/v1")}
    assert model_connection("openai_compatible", "studio/local-model").state == "unknown"


def test_hosted_custom_endpoints_are_checked_without_dns(isolated_engine, monkeypatch):
    isolated_engine.llm.endpoints = {"studio": CompatibleEndpoint(baseUrl="http://localhost:1234/v1")}
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    assert model_connection("openai_compatible", "studio/local-model").state == "needs_setup"
    isolated_engine.llm.endpoints = {"remote": CompatibleEndpoint(baseUrl="https://model.example.com/v1")}
    assert model_connection("openai_compatible", "remote/model").state == "unknown"
    missing = model_connection("openai", "gpt-6-sol")
    assert missing.state == "needs_setup" and missing.action.target == "environment"
    assert missing.action.key == "OPENAI_API_KEY"


def test_gateway_does_not_request_local_keys(monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", "https://gateway.example.com/v1")
    assert model_connection("openai", "gpt-6-sol").state == "configured"
    assert model_connection("gemini", "gemini-2.5-flash").state == "configured"
    assert model_connection("azure", "daily").state == "needs_setup"
    assert model_connection("gemini", "vertex_ai/gemini-2.5-flash").state == "needs_setup"
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", "https://private-secret@gateway.example.com")
    state = model_connection("openai", "gpt-6-sol")
    assert state.state == "needs_setup" and "private-secret" not in state.model_dump_json()


async def test_config_exposes_selected_setup_and_serializes_action(isolated_engine, monkeypatch):
    from rune.api.handlers.config import get_config_endpoint
    from rune.llm.reasoning import ReasoningControl

    # Existing /config capability lookup is unrelated to setup readiness.
    monkeypatch.setattr("rune.llm.reasoning.reasoning_control", lambda _: ReasoningControl())
    monkeypatch.setattr("rune.api.handlers.config._memory_tuning_state", lambda: {})
    isolated_engine.llm.active_provider = "anthropic"
    isolated_engine.llm.active_model = "claude-sonnet-5"
    data = json.loads((await get_config_endpoint()).model_dump_json(by_alias=True))
    assert data["modelConnection"]["provider"] == "anthropic"
    assert data["modelConnection"]["action"]["key"] == "ANTHROPIC_API_KEY"
    assert data["modelConnection"]["state"] == "needs_setup"


async def test_selectable_catalog_only_contains_providers_accepted_by_model_set(monkeypatch):
    from unittest.mock import AsyncMock

    from rune.llm.models import known_models, selectable_models
    from rune.types import Provider

    monkeypatch.setattr("rune.llm.endpoints.refresh_discovered", AsyncMock())
    monkeypatch.setattr("rune.llm.xai.available_model_ids", AsyncMock(return_value=None))
    selectable = await selectable_models()
    assert selectable and all(Provider(provider) for provider, _ in selectable)
    assert any(provider == "mistral" for provider, _ in known_models())
