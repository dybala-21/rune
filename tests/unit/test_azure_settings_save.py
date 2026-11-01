"""Save failures must not split a resource's key, endpoint, and deployments."""

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from rune.api.azure_settings import AzureSave, credential_revision, revision, save_settings
from rune.config import get_config

BASE = "https://fixture.services.ai.azure.com/openai/v1"
OTHER = "https://other.services.ai.azure.com/openai/v1"


@pytest.fixture
def files(tmp_path, monkeypatch):
    from rune.config.loader import reset_config

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.delenv("RUNE_CLOUD_WORKER", raising=False)
    for key in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    project = tmp_path / ".rune"
    project.mkdir()
    user_env = home / ".env"
    user_env.write_text('# keep this comment\nAZURE_API_KEY=old-secret\nOTHER_KEY="keep me"\n')
    project_env = project / ".env"
    project_env.write_text(f"AZURE_API_BASE={BASE}\n")
    config = project / "config.yaml"
    config.write_text('openai_api_key: ${OPENAI_API_KEY}\nllm:\n  azureDeployments:\n    daily: Kimi-K2.7-Code\n')
    for path in (user_env, project_env):
        path.chmod(0o600)
    config.chmod(0o644)
    monkeypatch.setenv("AZURE_API_BASE", BASE)
    monkeypatch.setenv("AZURE_API_KEY", "old-secret")
    reset_config()
    get_config()
    return [project_env, user_env, config]


def request(**updates):
    return AzureSave(**{"endpoint": OTHER, "apiKey": "new-secret", "revision": revision(),
                       "deployments": {"deep": "DeepSeek-V4-Pro"}, **updates})


def snapshot(files):
    return [(path.read_bytes(), path.stat().st_mode & 0o777) for path in files]


def assert_original(files, original):
    assert snapshot(files) == original
    assert os.environ["AZURE_API_BASE"] == BASE
    assert os.environ["AZURE_API_KEY"] == "old-secret"
    assert get_config().llm.azure_deployments == {"daily": "Kimi-K2.7-Code"}
    assert not any(files[0].parent.glob(".rune-settings-*"))
    assert not any(files[1].parent.glob(".rune-settings-*"))


def test_one_save_persists_connection_deployments_and_selection(files):
    from rune.config.loader import load_config

    before_revision = revision()
    before_credentials = credential_revision()
    result = save_settings(request(useDeployment="deep"))
    assert result["saved"] and result["activeDeployment"] == "deep"
    assert result["revision"] != before_revision and result["credentialRevision"] != before_credentials
    assert "new-secret" not in str(result)
    assert os.environ["AZURE_API_BASE"] == OTHER and os.environ["AZURE_API_KEY"] == "new-secret"
    assert get_config().llm.active_provider == "azure" and get_config().llm.active_model == "deep"
    assert '${OPENAI_API_KEY}' in files[2].read_text()
    assert "new-secret" not in files[2].read_text()
    assert '# keep this comment' in files[1].read_text() and 'OTHER_KEY="keep me"' in files[1].read_text()
    assert files[1].stat().st_mode & 0o777 == 0o600
    assert files[2].stat().st_mode & 0o777 == 0o644
    assert load_config(force=True).llm.active_model == "deep"
    assert not any(files[0].parent.glob(".rune-settings-*"))


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_failure_at_each_replacement_restores_every_file_and_runtime(files, monkeypatch, fail_at):
    from rune.config import transaction

    original = snapshot(files)
    replace = os.replace
    count = 0

    def failing(source, target):
        nonlocal count
        count += 1
        if count == fail_at:
            raise OSError("simulated write failure")
        return replace(source, target)

    monkeypatch.setattr(transaction.os, "replace", failing)
    with pytest.raises(HTTPException, match="original files"):
        save_settings(request(useDeployment="deep"))
    assert_original(files, original)


@pytest.mark.parametrize("fail_at", [1, 2, 4, 6])
def test_staging_failure_never_changes_destination_files(files, monkeypatch, fail_at):
    from rune.config import transaction

    original = snapshot(files)
    stage = transaction._stage
    count = 0

    def failing(*args):
        nonlocal count
        count += 1
        if count == fail_at:
            raise OSError("simulated full disk")
        return stage(*args)

    monkeypatch.setattr(transaction, "_stage", failing)
    with pytest.raises(HTTPException, match="original files"):
        save_settings(request())
    assert_original(files, original)


def test_changed_resource_requires_a_new_key_before_writes(files):
    original = snapshot(files)
    with pytest.raises(HTTPException, match="different resource"):
        save_settings(request(apiKey=None))
    assert_original(files, original)


def test_stale_tab_cannot_overwrite_a_successful_save(files):
    stale = request()
    save_settings(request(useDeployment="deep"))
    saved = snapshot(files)
    with pytest.raises(HTTPException, match="changed elsewhere") as exc:
        save_settings(stale)
    assert exc.value.status_code == 409 and snapshot(files) == saved


def test_key_rotation_changes_revision_but_plain_deployment_edit_keeps_credential_revision(files):
    initial = credential_revision()
    save_settings(request(endpoint=BASE, apiKey=None))
    assert credential_revision() == initial
    save_settings(request(endpoint=BASE))
    assert credential_revision() != initial


def test_process_owned_key_cannot_be_silently_overridden(files, monkeypatch):
    original = snapshot(files)
    monkeypatch.setenv("AZURE_API_KEY", "launch-secret")
    with pytest.raises(HTTPException, match="launch environment"):
        save_settings(request())
    assert snapshot(files) == original and os.environ["AZURE_API_KEY"] == "launch-secret"


def test_no_change_to_the_key_leaves_a_launch_owned_key_usable(files, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "launch-secret")
    save_settings(request(endpoint=BASE, apiKey=None))
    assert "old-secret" in files[1].read_text()
    assert os.environ["AZURE_API_KEY"] == "launch-secret"


def test_invalid_selection_and_pricing_fail_before_writes(files):
    original = snapshot(files)
    with pytest.raises(HTTPException, match="included"):
        save_settings(request(useDeployment="missing"))
    with pytest.raises(HTTPException, match="Pricing"):
        save_settings(request(pricing={"endpoint": BASE, "deployments": {"deep": "global"}}))
    assert_original(files, original)


def test_removing_active_deployment_requires_an_explicit_replacement(files):
    get_config().llm.active_provider = "azure"
    get_config().llm.active_model = "daily"
    with pytest.raises(HTTPException, match="active deployment was removed"):
        save_settings(request())
    save_settings(request(useDeployment="deep"))
    assert get_config().llm.active_model == "deep"


def test_rollback_failure_retains_protected_backup_and_reports_recovery(files, monkeypatch):
    from rune.config import transaction

    replace = os.replace
    count = 0

    def failing(source, target):
        nonlocal count
        count += 1
        if count >= 2:
            raise OSError("simulated filesystem unavailable")
        return replace(source, target)

    monkeypatch.setattr(transaction.os, "replace", failing)
    with pytest.raises(HTTPException, match="Saving and restoring settings failed"):
        save_settings(request())
    backups = list(files[0].parent.glob(".rune-settings-*"))
    assert backups and all(path.stat().st_mode & 0o777 == 0o600 for path in backups)
    assert os.environ["AZURE_API_KEY"] == "old-secret"
    assert get_config().llm.azure_deployments == {"daily": "Kimi-K2.7-Code"}


def test_new_file_is_removed_when_later_replacement_fails(files, monkeypatch):
    from rune.config.transaction import replace_settings

    new = files[0].parent / "new.env"
    original = files[2].read_bytes()
    replace = os.replace

    def failing(source, target):
        if Path(target) == files[2]:
            raise OSError("failed before replacing config")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError):
        replace_settings({new: ("test=value", 0o600), files[2]: ("changed", 0o644)})
    assert not new.exists() and files[2].read_bytes() == original


def test_symlinked_config_is_updated_without_replacing_the_link(files):
    original = files[2].read_bytes()
    target = files[2].parent / "actual-config.yaml"
    target.write_bytes(original)
    files[2].unlink()
    files[2].symlink_to(target)
    save_settings(request())
    assert files[2].is_symlink() and "DeepSeek-V4-Pro" in target.read_text()


def test_malformed_config_does_not_change_any_credentials(files):
    files[2].write_text("llm: [unterminated")
    original = snapshot(files)
    with pytest.raises(HTTPException, match="original files"):
        save_settings(request())
    assert_original(files, original)


def test_concurrent_saves_cannot_both_commit_the_same_revision(files):
    from concurrent.futures import ThreadPoolExecutor

    first, second = request(), request()

    def save(req):
        try:
            return save_settings(req)["saved"]
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, [first, second]))
    assert results.count(True) == 1 and results.count(409) == 1


def test_response_revision_belongs_to_this_save_even_when_another_save_follows(files, monkeypatch):
    state = {}

    def after_commit():
        if not state:
            state["first_revision"] = revision()
            save_settings(request(endpoint=OTHER, apiKey=None, deployments={"third": "Kimi-K2.7-Code"}, useDeployment="third"))

    monkeypatch.setattr("rune.llm.model_selection._reset_llm_client", after_commit)
    result = save_settings(request(useDeployment="deep"))
    assert result["revision"] == state["first_revision"] != revision()
    assert result["activeDeployment"] == "deep" and get_config().llm.active_model == "third"
