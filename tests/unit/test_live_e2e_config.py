"""E2E selection follows the chosen provider, including config-only credentials."""

import pytest

from rune.config.schema import RuneConfig
from rune.llm.models import PROVIDER_ENV_KEYS
from tests.e2e.live_config import configure_model, select_model


@pytest.fixture
def clean_env(monkeypatch):
    for key in {key for names in PROVIDER_ENV_KEYS.values() for key in names} | {"GOOGLE_APPLICATION_CREDENTIALS"}:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("provider,key", [("openai", "openai_api_key"), ("anthropic", "anthropic_api_key"), ("gemini", "gemini_api_key")])
def test_keys_from_config_are_valid_without_environment_variables(clean_env, provider, key):
    cfg = RuneConfig(**{key: "test-only-key"})
    assert select_model(cfg, provider, "test-model") == (provider, "test-model")


def test_other_providers_key_does_not_authorize_missing_provider(clean_env):
    cfg = RuneConfig(openai_api_key="test-only-key")
    with pytest.raises(ValueError, match="anthropic"):
        select_model(cfg, "anthropic")


def test_grok_environment_and_gemini_service_account(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("XAI_API_KEY", "test-only-key")
    cfg = RuneConfig()
    assert select_model(cfg, "xai")[0] == "xai"
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    cfg.google_credentials_file = str(credentials)
    assert select_model(cfg, "gemini")[0] == "gemini"
    credentials.unlink()
    with pytest.raises(ValueError, match="gemini"):
        select_model(cfg, "gemini")


def test_config_file_is_loaded_before_collection_checks_credentials(clean_env, monkeypatch, tmp_path):
    from rune.config import loader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("XAI_API_KEY=test-only-key\n")
    loader.reset_config()
    assert select_model(loader.get_config(), "xai")[0] == "xai"
    monkeypatch.delenv("XAI_API_KEY")


def test_model_override_is_local_to_the_copy(clean_env):
    original = RuneConfig(openai_api_key="test-only-key")
    cfg = original.model_copy(deep=True)
    configure_model(cfg, "openai", "chosen-model")
    assert select_model(cfg) == ("openai", "chosen-model")
    assert cfg.llm.models.openai.fast == "chosen-model"
    assert original.llm.active_model is None
def test_live_workflow_approves_only_the_named_fixture_edit(tmp_path):
    from tests.e2e.workflow_harness import Workflow

    work = tmp_path / "work"
    work.mkdir()
    harness = Workflow(None, work, "test", "test")
    run = {"approval": {"command": "file_edit"}, "toolCalls": [
        {"toolName": "file_edit", "args": {"path": str(work / "stats.py")}},
    ]}
    assert harness._can_approve_edit(run, ("stats.py",))
    assert not harness._can_approve_edit(run, ())
    run["approval"]["command"] = "bash_execute"
    assert not harness._can_approve_edit(run, ("stats.py",))
    run["approval"]["command"] = "file_edit"
    (work / "stats.py").symlink_to(tmp_path / "outside.py")
    assert not harness._can_approve_edit(run, ("stats.py",))
    run = {"approval": {"command": "file_write"}, "toolCalls": [
        {"toolName": "file_write", "args": {"path": str(work / "test_extra.py")}},
    ]}
    assert not harness._can_approve_edit(run, ())
    assert harness._can_approve_edit(run, (), allow_new_tests=True)
    (work / "test_extra.py").write_text("existing test")
    assert not harness._can_approve_edit(run, (), allow_new_tests=True)
