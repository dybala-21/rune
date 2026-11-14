"""Memory settings must be saved in a scope the engine will reload."""

import pytest

from rune.api.handlers.config import ConfigPatchRequest, patch_config
from rune.config.schema import RuneConfig


@pytest.mark.parametrize("hosted", [False, True])
async def test_project_memory_save_survives_reload_on_local_and_hosted_engines(hosted, monkeypatch, tmp_path):
    from rune.config import loader
    from rune.memory.tuning import MEMORY_TUNING_ENV_KEYS, get_tuning_config
    from rune.utils.env import _parse_env_file, load_env, project_env_path, user_env_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1" if hosted else "0")
    monkeypatch.setattr("rune.cloud.boundary._HOSTED", False)
    monkeypatch.setattr(loader, "_config", RuneConfig())
    for key in (*MEMORY_TUNING_ENV_KEYS.values(), "UNCHANGED_PROJECT_VALUE"):
        # Register even absent keys for cleanup: the real writer/loader mutates
        # os.environ without going through monkeypatch.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    project = project_env_path()
    project.parent.mkdir()
    prior_project = "RUNE_MEMORY_SEMANTIC_LIMIT=3\nUNCHANGED_PROJECT_VALUE=keep\n"
    project.write_text(prior_project)

    result = await patch_config(ConfigPatchRequest(memoryTuning={
        "scope": "project", "semanticLimit": 17, "semanticMinScore": 0.61,
    }))
    assert result.updated
    saved_path = user_env_path() if hosted else project
    saved = _parse_env_file(saved_path.read_text())
    assert saved["RUNE_MEMORY_SEMANTIC_LIMIT"] == "17"
    assert saved["RUNE_MEMORY_SEMANTIC_MIN_SCORE"] == "0.61"
    if hosted:
        assert project.read_text() == prior_project
    else:
        assert saved["UNCHANGED_PROJECT_VALUE"] == "keep"
        assert not user_env_path().exists()

    # Simulate a fresh engine's environment load, rather than accepting the
    # live os.environ value as evidence that the settings were persisted.
    for key in MEMORY_TUNING_ENV_KEYS.values():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("UNCHANGED_PROJECT_VALUE", raising=False)
    load_env()
    restored = get_tuning_config()
    assert restored["semantic_limit"] == 17
    assert restored["semantic_min_score"] == 0.61
