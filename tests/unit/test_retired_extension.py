"""The Browser Bridge extension an earlier version left behind: its commands are gone, and the
copy in RUNE's own folder is cleared away when RUNE next starts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from typer.testing import CliRunner

from rune.daemon import maintenance
from rune.daemon.main import RuneDaemon
from rune.daemon.maintenance import remove_retired_extension


def bridge(folder: Path, name: str = "RUNE Browser Bridge") -> Path:
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"manifest_version": 3, "name": name,
                                                      "permissions": ["debugger"]}))
    (folder / "background.js").write_text("// asks 127.0.0.1 for a relay")
    return folder


def test_the_folder_an_earlier_version_wrote_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    folder = bridge(tmp_path / "extension" / "rune-browser-bridge")
    (tmp_path / "config.yaml").write_text("kept")
    assert remove_retired_extension() is True
    assert not folder.exists() and not folder.parent.exists()
    assert (tmp_path / "config.yaml").read_text() == "kept"
    assert remove_retired_extension() is False, "nothing is left to remove"


def test_what_else_is_kept_beside_it_stays(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    folder = bridge(tmp_path / "extension" / "rune-browser-bridge")
    other = bridge(tmp_path / "extension" / "the-owners", name="The owner's own")
    assert remove_retired_extension() is True
    assert not folder.exists() and (other / "background.js").exists()


def test_a_folder_of_that_name_that_is_not_the_bridge_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    folder = bridge(tmp_path / "extension" / "rune-browser-bridge", name="The owner's own")
    assert remove_retired_extension() is False and (folder / "background.js").exists()
    for unread in ("not json", "[]", ""):
        (folder / "manifest.json").write_text(unread)
        assert remove_retired_extension() is False and (folder / "background.js").exists(), unread
    (folder / "manifest.json").unlink()
    assert remove_retired_extension() is False and (folder / "background.js").exists()


def test_a_link_to_the_bridge_is_taken_away_and_what_it_leads_to_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    real = bridge(tmp_path / "checkout" / "rune-browser-bridge")
    link = tmp_path / "home" / "extension" / "rune-browser-bridge"
    link.parent.mkdir(parents=True)
    link.symlink_to(real, target_is_directory=True)
    assert remove_retired_extension() is True
    assert not link.is_symlink() and (real / "manifest.json").exists()


async def test_the_daemon_clears_it_away_as_it_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    folder = bridge(tmp_path / "extension" / "rune-browser-bridge")
    daemon = RuneDaemon(install_signal_handlers=False)
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: SimpleNamespace())
    monkeypatch.setattr("rune.conversation.store.ConversationStore", lambda path: SimpleNamespace())
    monkeypatch.setattr(daemon, "_cleanup_orphaned_runs", AsyncMock())
    monkeypatch.setattr(maintenance, "recover_spend", AsyncMock())
    await daemon._init_subsystems()
    assert not folder.exists()


def test_the_commands_that_set_the_bridge_up_are_gone():
    from rune.cli.main import app

    for command in ("setup", "connect", "status"):
        assert CliRunner().invoke(app, ["browser", command]).exit_code == 2, command
