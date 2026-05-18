from __future__ import annotations

from rune.utils.paths import rune_data, rune_home, rune_logs


def test_rune_home_uses_rune_home_env(tmp_path, monkeypatch):
    custom_home = tmp_path / "bench-home"
    monkeypatch.setenv("RUNE_HOME", str(custom_home))

    assert rune_home() == custom_home
    assert custom_home.exists()


def test_rune_data_and_logs_follow_rune_home(tmp_path, monkeypatch):
    custom_home = tmp_path / "bench-home"
    monkeypatch.setenv("RUNE_HOME", str(custom_home))

    assert rune_data() == custom_home / "data"
    assert rune_logs() == custom_home / "logs"


def test_user_env_file_follows_rune_home(tmp_path, monkeypatch):
    custom_home = tmp_path / "bench-home"
    monkeypatch.setenv("RUNE_HOME", str(custom_home))

    from rune.utils.env import set_env

    set_env("RUNE_TEST_ENV_PATH", "ok", scope="user")
    monkeypatch.delenv("RUNE_TEST_ENV_PATH", raising=False)

    assert (custom_home / ".env").read_text(encoding="utf-8") == "RUNE_TEST_ENV_PATH=ok\n"


def test_file_trash_follows_rune_home(tmp_path, monkeypatch):
    custom_home = tmp_path / "bench-home"
    monkeypatch.setenv("RUNE_HOME", str(custom_home))

    from rune.tools.file import _trash_dir

    assert _trash_dir() == custom_home / "data" / "trash"
