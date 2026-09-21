"""Environment edits preserve unrelated entries, comments, and file permissions."""

from __future__ import annotations

import os

import pytest

from rune.utils.env import (
    _apply_env_edits,
    _parse_env_file,
    _serialize_env,
    _split_env_line,
    set_env,
    unset_env,
)


@pytest.fixture
def env_home(tmp_path, monkeypatch):
    home = tmp_path / "rune-home"
    home.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return home


SAMPLE = """\
# credentials
OPENAI_API_KEY=sk-abc

# search
BRAVE_API_KEY="bs-xyz"   # the key from the dashboard
export TELEGRAM_TOKEN=tg-1
QUOTED='single quoted'
EMPTY=
prose line with an = sign
"""


class TestParsing:
    def test_a_trailing_comment_is_not_part_of_the_value(self):
        assert _parse_env_file(SAMPLE)["BRAVE_API_KEY"] == "bs-xyz"

    def test_an_export_prefix_is_accepted(self):
        assert _parse_env_file(SAMPLE)["TELEGRAM_TOKEN"] == "tg-1"

    def test_single_quotes_are_literal_as_in_the_shell(self):
        assert _parse_env_file(r"X='a\b'")["X"] == r"a\b"

    def test_an_empty_value_is_kept(self):
        assert _parse_env_file(SAMPLE)["EMPTY"] == ""

    def test_a_prose_line_is_not_mistaken_for_a_variable(self):
        assert "prose line with an " not in _parse_env_file(SAMPLE)

    def test_an_unterminated_quote_takes_the_rest_of_the_line(self):
        assert _parse_env_file('X="no closing quote')["X"] == "no closing quote"

    @pytest.mark.parametrize(
        "value",
        [
            'has "quotes" and spaces',
            "plain",
            "trail#hash",
            r"back\slash",
            r'both "q" and \b',
            "",
            "  padded  ",
            "single ' quote",
        ],
    )
    def test_every_value_survives_a_serialize_parse_round_trip(self, value):
        assert _parse_env_file(_serialize_env({"K": value}))["K"] == value


class TestSurgicalEdits:
    def test_comments_and_unrelated_lines_survive(self):
        out = _apply_env_edits(SAMPLE, {"BRAVE_API_KEY": "bs-new"})

        assert "# credentials" in out
        assert "# search" in out
        assert "prose line with an = sign" in out
        assert "OPENAI_API_KEY=sk-abc" in out
        assert "BRAVE_API_KEY=bs-new" in out

    def test_a_new_key_is_appended(self):
        out = _apply_env_edits(SAMPLE, {"RUNE_NEW": "v"})
        assert out.rstrip().endswith("RUNE_NEW=v")

    def test_none_removes_the_line(self):
        out = _apply_env_edits(SAMPLE, {"TELEGRAM_TOKEN": None})
        assert "TELEGRAM_TOKEN" not in out
        assert "OPENAI_API_KEY=sk-abc" in out

    def test_rewriting_the_same_value_is_stable(self):
        once = _apply_env_edits(SAMPLE, {"BRAVE_API_KEY": "bs-xyz"})
        twice = _apply_env_edits(once, {"BRAVE_API_KEY": "bs-xyz"})
        assert once == twice

    def test_removing_the_last_variable_empties_the_file(self):
        assert _apply_env_edits("ONLY=1\n", {"ONLY": None}) == ""


class TestWritesThroughTheFile:
    def test_set_then_unset_leaves_the_file_byte_identical(self, env_home):
        path = env_home / ".env"
        path.write_text(SAMPLE)
        before = path.read_bytes()

        set_env("RUNE_PROBE", "1", scope="user")
        assert "RUNE_PROBE=1" in path.read_text()

        unset_env("RUNE_PROBE", scope="user")
        assert path.read_bytes() == before

    def test_the_file_is_written_owner_only(self, env_home):
        set_env("RUNE_PROBE", "1", scope="user")
        assert (env_home / ".env").stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize("scope", ["user", "project"])
    def test_effective_key_replacement_survives_a_restart(self, env_home, monkeypatch, scope):
        from rune.utils.env import effective_env_scope, list_env, load_env

        key = "TYPESAFE_API_KEY"
        monkeypatch.delenv(key, raising=False)
        set_env(key, "dummy-user-value", scope="user")
        if scope == "project":
            set_env(key, "dummy-project-value", scope="project")
        assert effective_env_scope(key) == scope
        set_env(key, "dummy-replacement-value", scope="effective")
        if scope == "project":
            assert list_env()["user"][key] == "dummy-user-value"
        monkeypatch.delenv(key)
        load_env()
        assert os.environ[key] == "dummy-replacement-value"
        assert effective_env_scope(key) == scope

    async def test_launch_environment_override_is_not_reported_as_a_saved_replacement(self, env_home, monkeypatch):
        from fastapi import HTTPException

        from rune.api.handlers.env import EnvSetRequest
        from rune.api.handlers.env import set_env as save
        from rune.utils.env import effective_env_scope

        key = "TYPESAFE_API_KEY"
        monkeypatch.delenv(key, raising=False)
        set_env(key, "dummy-file-value", scope="user")
        before = (env_home / ".env").read_bytes()
        monkeypatch.setenv(key, "dummy-launch-value")
        assert effective_env_scope(key) == "process"
        with pytest.raises(HTTPException) as error:
            await save(key, EnvSetRequest(value="dummy-replacement", scope="effective"))
        assert error.value.status_code == 409
        assert (env_home / ".env").read_bytes() == before
        assert os.environ[key] == "dummy-launch-value"

    def test_an_updated_key_keeps_its_position(self, env_home):
        path = env_home / ".env"
        path.write_text(SAMPLE)

        set_env("OPENAI_API_KEY", "sk-new", scope="user")

        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines[0] == "# credentials"
        assert lines[1] == "OPENAI_API_KEY=sk-new"


class TestOneParserForBothReaders:
    """The config loader and settings editor share the same parsing rules."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ('BRAVE_API_KEY="bs-real"   # from dashboard', ("BRAVE_API_KEY", "bs-real")),
            ("export TELEGRAM_TOKEN=tg-1", ("TELEGRAM_TOKEN", "tg-1")),
            ("HASHY=abc #comment", ("HASHY", "abc")),
            ('ESCAPED="say \\"hi\\""', ("ESCAPED", 'say "hi"')),
            ("this is prose = with equals", None),
        ],
    )
    def test_the_loader_agrees_with_this_module(self, line, expected):
        from rune.config.loader import _load_dotenv  # noqa: F401  (import path check)

        assert _split_env_line(line) == expected

    def test_load_dotenv_sets_the_real_name_and_no_bogus_ones(self, env_home, monkeypatch):
        from rune.config.loader import _load_dotenv

        (env_home / ".env").write_text(
            'BRAVE_API_KEY="bs-real"   # note\n'
            "export TELEGRAM_TOKEN=tg-1\n"
            "this is prose = with equals\n"
        )
        for key in ("BRAVE_API_KEY", "TELEGRAM_TOKEN"):
            monkeypatch.delenv(key, raising=False)

        _load_dotenv()

        assert os.environ["BRAVE_API_KEY"] == "bs-real"
        assert os.environ["TELEGRAM_TOKEN"] == "tg-1"
        assert "export TELEGRAM_TOKEN" not in os.environ
        assert "this is prose" not in os.environ


class TestConcurrentEnvWrites:
    """Concurrent edits preserve every key and the existing file contents."""

    def test_overlapping_writes_keep_every_variable(self, env_home):
        import threading

        path = env_home / ".env"
        path.write_text("# keep me\nAPI_KEY=sk-secret\n")

        threads = [
            threading.Thread(target=set_env, args=(f"RUNE_K{i}", str(i)), kwargs={"scope": "user"})
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        written = _parse_env_file(path.read_text())
        assert {f"RUNE_K{i}" for i in range(20)} <= set(written)
        assert written["API_KEY"] == "sk-secret"
        assert "# keep me" in path.read_text()

    def test_the_file_is_replaced_not_truncated(self, env_home):
        """A crash mid-write must not leave a half-written credentials file."""
        path = env_home / ".env"
        path.write_text("API_KEY=sk-secret\n")

        set_env("RUNE_PROBE", "1", scope="user")

        assert not [p for p in env_home.iterdir() if ".tmp" in p.name]
        assert path.stat().st_mode & 0o777 == 0o600
