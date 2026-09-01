"""Settings written from the UI have to survive a restart.

Every handler here used to change memory and report success, so a toggle looked
like it applied and was gone on the next launch. These tests pin the write path:
the file is patched surgically, unrelated keys and comments survive, and secrets
never get rewritten.
"""

from __future__ import annotations

import os

import pytest

from rune.config.loader import _unknown_keys
from rune.config.schema import RuneConfig
from rune.config.writer import _assign, save_config_values


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Point config reads and writes at a throwaway directory."""
    home = tmp_path / "rune-home"
    home.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.chdir(tmp_path)  # keep the project-level lookup out of the repo

    from rune.config import loader

    monkeypatch.setattr(loader, "_config", None)
    monkeypatch.setattr(loader, "_config_mtime", 0.0)
    return home


SAMPLE = """\
# RUNE configuration
llm:
  defaultProvider: openai
  # a provider block the schema does not model
  providers:
    ollama:
      baseUrl: http://localhost:11434
  activeModel: old-model
proactive:
  enabled: true
"""


class TestTheWriterOnlyTouchesWhatItIsAsked:
    def test_unknown_keys_and_comments_survive_a_write(self, config_home):
        path = config_home / "config.yaml"
        path.write_text(SAMPLE)

        save_config_values({"llm.activeModel": "new-model"})

        after = path.read_text()
        assert "new-model" in after
        assert "old-model" not in after
        # The schema models none of these; a full round-trip would drop them.
        assert "providers:" in after
        assert "baseUrl: http://localhost:11434" in after
        assert "# RUNE configuration" in after
        assert "# a provider block the schema does not model" in after

    def test_a_none_value_removes_the_key(self, config_home):
        path = config_home / "config.yaml"
        path.write_text(SAMPLE)

        save_config_values({"llm.activeModel": None})

        assert "activeModel" not in path.read_text()
        assert "defaultProvider: openai" in path.read_text()

    def test_it_creates_the_file_when_there_is_none(self, config_home):
        path = config_home / "config.yaml"
        assert not path.exists()

        assert save_config_values({"llm.activeModel": "m"}) == path
        assert "activeModel: m" in path.read_text()

    def test_nested_paths_are_created_on_demand(self, config_home):
        (config_home / "config.yaml").write_text("llm:\n  defaultProvider: openai\n")

        save_config_values({"a.b.c": 1})

        assert "c: 1" in (config_home / "config.yaml").read_text()

    def test_an_env_placeholder_is_not_replaced_with_its_secret(self, config_home):
        """The loader resolves ``${VAR}`` in memory; the file must keep the placeholder."""
        path = config_home / "config.yaml"
        path.write_text("openai_api_key: ${OPENAI_API_KEY}\nllm:\n  activeModel: a\n")
        os.environ["OPENAI_API_KEY"] = "sk-should-never-be-written"

        save_config_values({"llm.activeModel": "b"})

        assert "${OPENAI_API_KEY}" in path.read_text()
        assert "sk-should-never-be-written" not in path.read_text()

    def test_it_leaves_no_temp_file_behind(self, config_home):
        (config_home / "config.yaml").write_text(SAMPLE)

        save_config_values({"llm.activeModel": "x"})

        assert [p.name for p in config_home.iterdir()] == ["config.yaml"]

    def test_an_empty_update_writes_nothing(self, config_home):
        assert save_config_values({}) is None


class TestAWriteIsVisibleToTheNextRead:
    """The daemon re-reads config on a timer; a stale cache made it blind."""

    def test_the_next_load_sees_what_was_just_written(self, config_home):
        from rune.config import load_config

        (config_home / "config.yaml").write_text("proactive:\n  enabled: true\n")
        load_config(force=True)
        assert load_config().proactive.enabled is True

        save_config_values({"proactive.enabled": False})

        assert load_config().proactive.enabled is False

    def test_a_session_only_override_is_not_clobbered_by_a_write(self, config_home):
        """get_config() keeps the cached object, so a temporary switch survives."""
        from rune.config import get_config, load_config

        (config_home / "config.yaml").write_text("llm:\n  defaultProvider: openai\n")
        load_config(force=True)
        get_config().llm.active_model = "temporary-escalation"

        save_config_values({"proactive.enabled": False})

        assert get_config().llm.active_model == "temporary-escalation"


class TestConcurrentWrites:
    def test_no_update_is_lost_when_writes_overlap(self, config_home):
        import threading

        (config_home / "config.yaml").write_text("llm:\n  defaultProvider: openai\n")

        threads = [
            threading.Thread(target=save_config_values, args=({f"stress.k{i}": i},))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from ruamel.yaml import YAML

        data = YAML().load(config_home / "config.yaml")
        assert sorted(data["stress"]) == sorted(f"k{i}" for i in range(20))
        assert data["llm"]["defaultProvider"] == "openai"

    def test_no_temp_file_survives_the_storm(self, config_home):
        import threading

        threads = [
            threading.Thread(target=save_config_values, args=({f"a.k{i}": i},))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert [p.name for p in config_home.iterdir()] == ["config.yaml"]


class TestTheFileKeepsItsIdentity:
    def test_existing_permissions_are_kept(self, config_home):
        path = config_home / "config.yaml"
        path.write_text(SAMPLE)
        os.chmod(path, 0o644)

        save_config_values({"llm.activeModel": "x"})

        assert path.stat().st_mode & 0o777 == 0o644

    def test_a_symlinked_config_stays_a_symlink(self, config_home):
        """A config linked into a dotfiles repo must not be swapped for a file."""
        real = config_home / "dotfiles" / "config.yaml"
        real.parent.mkdir()
        real.write_text(SAMPLE)
        link = config_home / "config.yaml"
        link.symlink_to(real)

        save_config_values({"llm.activeModel": "linked"})

        assert link.is_symlink()
        assert "linked" in real.read_text()


class TestAssign:
    def test_it_replaces_a_non_dict_parent(self):
        root: dict = {"llm": "not-a-dict"}
        _assign(root, "llm.model", "m")
        assert root == {"llm": {"model": "m"}}

    def test_removing_through_a_missing_parent_is_a_no_op(self):
        root: dict = {}
        _assign(root, "nope.gone", None)
        assert root == {}


class TestIgnoredKeysAreReported:
    """Pydantic drops unknown keys silently, which reads as 'the setting applied'."""

    def test_it_names_the_keys_the_schema_will_drop(self):
        raw = {
            "llm": {"defaultProvider": "openai", "madeUpKey": 1},
            "safety": {"rolloutMode": "strict"},
            "nonsense": True,
        }

        found = _unknown_keys(raw, RuneConfig)

        assert "llm.madeUpKey" in found
        assert "nonsense" in found
        assert "llm.defaultProvider" not in found

    def test_a_fully_known_config_reports_nothing(self):
        assert _unknown_keys({"llm": {"defaultProvider": "openai"}}, RuneConfig) == []

    def test_it_survives_a_scalar_where_a_section_was_expected(self):
        assert _unknown_keys({"llm": "oops"}, RuneConfig) == []


class TestTheProactiveToggleReachesTheDaemon:
    """The daemon kept its own settings dict and hardcoded this to True."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_the_daemon_reads_what_the_file_says(self, config_home, enabled):
        from rune.config import load_config
        from rune.daemon.main import _configured_proactive_enabled

        save_config_values({"proactive.enabled": enabled})
        load_config(force=True)

        assert _configured_proactive_enabled() is enabled

    def test_an_unreadable_config_keeps_proactive_on(self, config_home, monkeypatch):
        from rune.daemon import main as daemon_main

        def boom():
            raise OSError("disk gone")

        monkeypatch.setattr("rune.config.load_config", boom)
        assert daemon_main._configured_proactive_enabled() is True


class TestModelSelectionPersists:
    def test_picking_a_model_writes_it_to_the_file(self, config_home, monkeypatch):
        from rune.llm.model_selection import (
            ActiveModelSelection,
            persist_active_model_selection,
        )
        from rune.types import Provider

        monkeypatch.setattr(
            "rune.llm.model_selection._reset_llm_client", lambda: None
        )
        persist_active_model_selection(
            ActiveModelSelection(provider=Provider.OPENAI, model="gpt-5.4")
        )

        written = (config_home / "config.yaml").read_text()
        assert "activeProvider: openai" in written
        assert "activeModel: gpt-5.4" in written

    def test_clearing_removes_the_override(self, config_home, monkeypatch):
        from rune.llm.model_selection import clear_active_model_selection

        (config_home / "config.yaml").write_text(
            "llm:\n  activeProvider: openai\n  activeModel: gpt-5.4\n"
        )
        monkeypatch.setattr(
            "rune.llm.model_selection._reset_llm_client", lambda: None
        )
        clear_active_model_selection()

        written = (config_home / "config.yaml").read_text()
        assert "activeProvider" not in written
        assert "activeModel" not in written


class TestPatchValidatesBeforeApplying:
    """Applying field by field left earlier fields live in memory when a later
    one was rejected: the client saw an error while the agent had switched
    models."""

    @pytest.fixture
    def patched_config(self, config_home):
        from rune.config import load_config

        (config_home / "config.yaml").write_text(
            "llm:\n  defaultProvider: openai\n  defaultModel: gpt-5.4\n"
        )
        load_config(force=True)
        return config_home

    @pytest.mark.asyncio
    async def test_a_rejected_request_leaves_the_running_model_alone(
        self, patched_config
    ):
        from fastapi import HTTPException

        from rune.api.handlers.config import ConfigPatchRequest, patch_config
        from rune.config import get_config

        with pytest.raises(HTTPException) as exc:
            await patch_config(
                ConfigPatchRequest(
                    activeModel={"provider": "ollama", "model": "qwen3"},
                    safetyTuning={"preset": "conservative"},
                )
            )

        assert exc.value.status_code == 501
        assert get_config().llm.default_model == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_an_unknown_provider_never_reaches_the_file(self, patched_config):
        from fastapi import HTTPException

        from rune.api.handlers.config import ConfigPatchRequest, patch_config

        with pytest.raises(HTTPException) as exc:
            await patch_config(
                ConfigPatchRequest(activeModel={"provider": "opemai", "model": "x"})
            )

        assert exc.value.status_code == 400
        assert "opemai" not in (patched_config / "config.yaml").read_text()

    @pytest.mark.asyncio
    async def test_a_bad_policy_mode_writes_no_env_var(self, patched_config, tmp_path):
        from fastapi import HTTPException

        from rune.api.handlers.config import ConfigPatchRequest, patch_config

        with pytest.raises(HTTPException) as exc:
            await patch_config(
                ConfigPatchRequest(
                    memoryTuning={
                        "scope": "project",
                        "policyMode": "auto",
                        "uncertainSemanticLimit": 7,
                    }
                )
            )

        assert exc.value.status_code == 400
        assert not (tmp_path / ".rune" / ".env").exists()

