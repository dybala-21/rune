"""Environment persistence feedback reflects known consumers, not key validation."""

import os

import pytest

from rune.api.env_activation import env_activation, provider_for_env_key
from rune.api.handlers.env import EnvSetRequest, delete_env, set_env


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.mark.parametrize(("key", "provider"), [
    ("OPENAI_API_KEY", "openai"), ("ANTHROPIC_API_KEY", "anthropic"),
    ("XAI_API_KEY", "xai"), ("GEMINI_API_KEY", "gemini"), ("GOOGLE_API_KEY", "gemini"),
    ("AZURE_OPENAI_API_KEY", "azure"), ("AZURE_API_KEY", "azure"),
    ("AZURE_OPENAI_ENDPOINT", "azure"), ("AZURE_API_BASE", "azure"),
    ("OLLAMA_HOST", "ollama"), ("OLLAMA_API_BASE", "ollama"),
    ("VERTEX_LOCATION", "gemini"), ("VERTEX_PROJECT", "gemini"),
    ("RUNE_ENDPOINT_STUDIO_API_KEY", "openai_compatible"),
])
def test_known_model_inputs_are_read_for_new_runs(key, provider):
    assert env_activation(key).state == "next_run"
    assert provider_for_env_key(key) == provider


@pytest.mark.parametrize("key", ["GOOGLE_APPLICATION_CREDENTIALS", "BRAVE_API_KEY", "RUNE_LOG_LEVEL",
                                 "RUNE_LOG_FILE", "TELEGRAM_BOT_TOKEN", "SLACK_BOT_TOKEN"])
def test_cached_consumers_require_restart(key):
    assert env_activation(key).state == "restart_required"


def test_unrecognized_setting_has_no_invented_activation_contract():
    assert env_activation("OPENAI_UNKNOWN_OPTION").state == "unknown"
    assert provider_for_env_key("MY_CUSTOM_KEY") is None


async def test_saved_key_is_used_by_new_run_but_existing_run_keeps_snapshot(isolated_env):
    from rune.llm.context import model_scope
    from rune.llm.target import resolve_target

    first = await set_env("OPENAI_API_KEY", EnvSetRequest(value="secret-before", scope="user"))
    assert first.updated and first.activation.state == "next_run" and first.provider == "openai"
    with model_scope():
        assert dict(resolve_target("gpt-6-sol").options)["api_key"] == "secret-before"
        changed = await set_env("OPENAI_API_KEY", EnvSetRequest(value="secret-after", scope="user"))
        assert dict(resolve_target("gpt-6-sol").options)["api_key"] == "secret-before"
    with model_scope():
        assert dict(resolve_target("gpt-6-sol").options)["api_key"] == "secret-after"
    assert os.environ["OPENAI_API_KEY"] == "secret-after"
    assert "secret-after" not in changed.model_dump_json()
    deleted = await delete_env("OPENAI_API_KEY", scope="user")
    assert deleted.deleted and deleted.activation.state == "next_run" and deleted.provider == "openai"
    assert "OPENAI_API_KEY" not in os.environ
    with model_scope():
        assert "api_key" not in dict(resolve_target("gpt-6-sol").options)


async def test_failed_persistence_does_not_report_success_or_change_live_key(isolated_env, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setenv("ANTHROPIC_API_KEY", "still-active")

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr("rune.utils.env._write_sensitive_env_file", fail)
    with pytest.raises(HTTPException) as exc:
        await set_env("ANTHROPIC_API_KEY", EnvSetRequest(value="replacement", scope="user"))
    assert exc.value.status_code == 500
    assert os.environ["ANTHROPIC_API_KEY"] == "still-active"


async def test_save_and_delete_unknown_setting_both_report_unknown(isolated_env):
    created = await set_env("RUNE_CUSTOM_SETTING", EnvSetRequest(value="value", scope="project"))
    assert created.activation.state == "unknown" and created.provider is None
    deleted = await delete_env("RUNE_CUSTOM_SETTING", scope="project")
    assert deleted.activation.state == "unknown" and deleted.provider is None


async def test_hosted_user_scope_is_persistent_but_project_feedback_warns(isolated_env, monkeypatch):
    from rune.utils.env import list_env

    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    saved = await set_env("OPENAI_API_KEY", EnvSetRequest(value="user-key", scope="user"))
    assert saved.activation.state == "next_run"
    assert list_env()["user"]["OPENAI_API_KEY"] == "user-key"
    project = await set_env("OPENAI_API_KEY", EnvSetRequest(value="project-key", scope="project"))
    assert project.activation.state == "unknown"
    assert "ignore project" in project.activation.reason
    assert list_env()["project"] == {}
    assert os.environ["OPENAI_API_KEY"] == "project-key"
    effective = await set_env("OPENAI_API_KEY", EnvSetRequest(value="effective-key", scope="effective"))
    assert effective.activation.state == "unknown"
    assert "effective-key" not in effective.model_dump_json()


async def test_gemini_alias_changes_keep_inflight_credentials_stable(isolated_env, monkeypatch):
    from rune.llm.context import model_scope
    from rune.llm.target import resolve_target

    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
                "RUNE_LLM_GATEWAY_URL"):
        monkeypatch.delenv(key, raising=False)
    await set_env("GOOGLE_API_KEY", EnvSetRequest(value="alias-before", scope="user"))
    with model_scope():
        assert dict(resolve_target("gemini/gemini-2.5-flash").options)["api_key"] == "alias-before"
        changed = await set_env("GOOGLE_API_KEY", EnvSetRequest(value="alias-after", scope="user"))
        assert changed.activation.state == "next_run" and changed.provider == "gemini"
        assert dict(resolve_target("gemini/gemini-2.5-flash").options)["api_key"] == "alias-before"
    with model_scope():
        assert dict(resolve_target("gemini/gemini-2.5-flash").options)["api_key"] == "alias-after"
    monkeypatch.setenv("GEMINI_API_KEY", "canonical-key")
    with model_scope():
        assert dict(resolve_target("gemini/gemini-2.5-flash").options)["api_key"] == "canonical-key"


@pytest.mark.parametrize("key", ["MISTRAL_API_KEY", "DEEPSEEK_API_KEY", "COHERE_API_KEY"])
def test_provider_handoff_does_not_offer_unsupported_selection(key):
    assert provider_for_env_key(key) is None


def test_alias_resolution_preserves_azure_vertex_and_gateway_precedence(isolated_env, monkeypatch, tmp_path):
    from rune.llm.target import resolve_target

    monkeypatch.delenv("RUNE_LLM_GATEWAY_URL", raising=False)
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "azure-canonical")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-alias")
    assert dict(resolve_target("azure/daily").options)["api_key"] == "azure-canonical"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-canonical")
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-alias")
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    vertex = resolve_target("gemini/gemini-2.5-flash")
    assert vertex.wire_model.startswith("vertex_ai/") and "api_key" not in dict(vertex.options)
    monkeypatch.setenv("RUNE_LLM_GATEWAY_URL", "https://gateway.example.com")
    monkeypatch.setenv("RUNE_LLM_GATEWAY_KEY", "gateway-credential")
    routed = resolve_target("gemini/gemini-2.5-flash")
    assert routed.wire_model.startswith("gemini/")
    assert dict(routed.options)["api_key"] == "gateway-credential"
