"""Gateway mode: no provider key leaves the machine, unrouted providers fail closed, failures stay visible."""

import pytest

from rune.agent.failover import LLMProfile, classify_error, determine_strategy
from rune.llm.gateway import GatewayConfigError, gateway_active, is_gateway_base
from rune.llm.pricing import request_price_known
from rune.llm.target import resolve_target

GATEWAY = "https://llm.gateway.internal"


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", GATEWAY + "/")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-local-key-must-not-be-sent")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-key-must-not-be-sent")
    monkeypatch.delenv("RUNE_LLM_GATEWAY_KEY", raising=False)


def test_off_by_default_keeps_direct_provider_options(monkeypatch):
    monkeypatch.delenv("RUNE_LLM_GATEWAY_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct")
    options = dict(resolve_target("anthropic/claude-sonnet-5").options)
    assert options == {"api_key": "sk-ant-direct"}
    assert not gateway_active()


@pytest.mark.parametrize(("model", "provider", "wire"), [
    ("anthropic/claude-sonnet-5", "anthropic", "anthropic/claude-sonnet-5"),
    ("claude-sonnet-5", "anthropic", "anthropic/claude-sonnet-5"),
    ("gpt-5.6", "openai", "gpt-5.6"),
    ("xai/grok-4.7", "xai", "xai/grok-4.7"),
])
def test_routed_providers_use_the_gateway_and_never_the_local_key(gateway, model, provider, wire):
    target = resolve_target(model)
    options = dict(target.options)
    assert target.wire_model == wire
    assert options["api_base"] == f"{GATEWAY}/{provider}"
    assert options["api_key"] == "rune-gateway"
    assert "must-not-be-sent" not in repr(target.kwargs())


def test_scoped_gateway_key_replaces_the_placeholder(gateway, monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_KEY", "vk-tenant-7")
    assert dict(resolve_target("claude-sonnet-5").options)["api_key"] == "vk-tenant-7"


def test_gemini_skips_vertex_credentials_under_the_gateway(gateway, monkeypatch, tmp_path):
    credentials = tmp_path / "sa.json"
    credentials.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    target = resolve_target("gemini/gemini-3.1-pro-preview")
    assert target.wire_model == "gemini/gemini-3.1-pro-preview"
    assert "vertex_project" not in dict(target.options) and "vertex_location" not in dict(target.options)


@pytest.mark.parametrize("model", ["ollama/qwen3", "azure/my-deploy", "vertex_ai/gemini-3.1-pro-preview",
                                   "groq/llama-4", "openai_compatible/box/model"])
def test_unrouted_providers_fail_closed(gateway, model):
    with pytest.raises(GatewayConfigError, match="not available through the LLM gateway"):
        resolve_target(model)


@pytest.mark.parametrize("url", ["http://llm.example.com", "http://8.8.8.8", "ftp://10.0.0.1",
                                 "https://user:pw@llm.example.com", "https://llm.example.com?x=1"])
def test_misconfigured_gateway_refuses_instead_of_using_direct_keys(url, monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct")
    with pytest.raises(GatewayConfigError):
        resolve_target("claude-sonnet-5")
    assert gateway_active()


def test_private_http_gateway_on_a_host_only_link(monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", "http://10.207.0.1:4000")
    assert dict(resolve_target("claude-sonnet-5").options)["api_base"] == "http://10.207.0.1:4000/anthropic"


def test_gateway_requests_keep_list_prices_for_cost_limits(gateway):
    params = {**resolve_target("anthropic/claude-sonnet-5").kwargs(), "_rune_source_model": "anthropic/claude-sonnet-5"}
    assert request_price_known(params)
    other = {**params, "api_base": "https://elsewhere.example.com/anthropic"}
    assert not request_price_known(other)
    assert is_gateway_base(GATEWAY + "/openai") and not is_gateway_base(GATEWAY + "evil.example/openai")


def _profiles():
    return [LLMProfile(name="a", provider="anthropic", model="claude-sonnet-5", priority=0),
            LLMProfile(name="b", provider="openai", model="gpt-5.6", priority=1)]


@pytest.mark.parametrize("reason", ["auth", "billing"])
def test_gateway_refusals_abort_instead_of_switching_models(gateway, reason):
    profiles = _profiles()
    assert determine_strategy(reason, profiles[0], 3, profiles).action == "abort"


def test_direct_mode_still_switches_on_auth(monkeypatch):
    monkeypatch.delenv("RUNE_LLM_GATEWAY_URL", raising=False)
    profiles = _profiles()
    assert determine_strategy("auth", profiles[0], 3, profiles).action == "switch_profile"


def test_gateway_403_is_a_policy_refusal(gateway, monkeypatch):
    assert classify_error("Error code: 403 - tenant suspended") == "auth"
    monkeypatch.delenv("RUNE_LLM_GATEWAY_URL")
    assert classify_error("Error code: 403 - tenant suspended") != "auth"


def test_model_picker_lists_only_routed_providers(gateway):
    from rune.llm.models import _has_provider_key

    assert _has_provider_key("anthropic") and _has_provider_key("mistral")
    assert not _has_provider_key("azure")


# --- review follow-ups: paths outside resolve_target, error classification, URL edge cases ---

@pytest.mark.parametrize("url", ["https://gw.example#", "https://gw.example?", "https://gw.example:99999",
                                 "http://[::1", "http://169.254.169.254", "http://0.0.0.0:4000",
                                 "http://[fe80::1]:4000", "https://169.254.169.254", "https://gw example.com"])
def test_service_url_edge_cases_are_refused(url):
    from rune.utils.service_url import service_url

    with pytest.raises(ValueError):
        service_url(url, allow_path=True)


@pytest.mark.parametrize("url", ["https://gw.example", "HTTPS://gw.example/v1/", "http://10.0.0.5:4000", "http://[fd00::5]:4000"])
def test_service_url_accepts_usable_endpoints(url):
    from rune.utils.service_url import service_url

    assert service_url(url, allow_path=True) == url.rstrip("/")


def test_malformed_gateway_never_crashes_error_handling(monkeypatch):
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", "http://[::1")
    assert gateway_active()
    assert classify_error(Exception("boom")) == "unknown"
    with pytest.raises(GatewayConfigError):
        resolve_target("claude-sonnet-5")


def test_malformed_broker_url_hides_connectors_without_crashing(monkeypatch):
    from rune.connectors.client import broker_available

    monkeypatch.setenv("RUNE_CONNECTOR_URL", "http://[::1")
    assert broker_available() is False


def test_gateway_classification_uses_status_not_substrings(gateway):
    class Refused(Exception):
        status_code = 403

    assert classify_error("prompt is 1403 tokens too long") != "auth"
    assert classify_error(Refused("denied")) == "auth"
    assert classify_error("Error code: 429 - You exceeded your current quota") == "billing"
    assert classify_error("Error code: 429 - slow down") == "rate_limit"


async def test_gateway_mode_never_lists_models_with_local_keys(gateway, monkeypatch):
    from rune.llm import models

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("a direct provider call was made in gateway mode")

    monkeypatch.setattr(models, "_cached_models", None)
    monkeypatch.setattr(models, "_fetch_openai_models", forbidden)
    monkeypatch.setattr("rune.llm.xai.available_model_ids", forbidden)
    monkeypatch.setattr("rune.llm.endpoints.refresh_discovered", forbidden)
    listed = await models.get_available_models()
    assert listed and {m.provider for m in listed} <= {"openai", "anthropic", "gemini", "xai", "mistral", "deepseek", "cohere"}
    assert all(provider not in {"azure", "ollama", "openai_compatible"} for provider, _ in models.known_models())
    monkeypatch.setattr(models, "_cached_models", None)


def test_gateway_mode_disables_third_party_decision_backends(gateway, monkeypatch):
    from rune.agent.decision_router import accelerator_status

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-local-key")
    assert accelerator_status(backend="jev") == "disabled"


def test_gateway_mode_offers_no_cloud_speech_to_text(gateway, monkeypatch):
    from rune.voice import service

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-local")
    monkeypatch.setenv("GROQ_API_KEY", "gq-local")
    provider = service._auto_detect_stt()
    assert provider is None or type(provider).__name__.lower().startswith("sherpa")
