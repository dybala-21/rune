"""Tests for the shared typed env getters in ``rune.utils.env``.

These back the F1 refactor that collapsed five duplicated ``_env_flag`` /
``_env_int`` definitions across the agent/bench/capability modules into one
source of truth. The delegating modules import these by alias, so behaviour
changes here would silently shift benchmark gating everywhere.
"""

from __future__ import annotations

import pytest

from rune.utils.env import env_flag, env_int


class TestEnvFlag:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On", " yes "])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("RUNE_TEST_FLAG", value)
        assert env_flag("RUNE_TEST_FLAG") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "2", "enabled"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("RUNE_TEST_FLAG", value)
        assert env_flag("RUNE_TEST_FLAG") is False

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("RUNE_TEST_FLAG", raising=False)
        assert env_flag("RUNE_TEST_FLAG") is False


class TestEnvInt:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("RUNE_TEST_INT", raising=False)
        assert env_int("RUNE_TEST_INT", 42) == 42

    def test_unset_without_default_returns_none(self, monkeypatch):
        monkeypatch.delenv("RUNE_TEST_INT", raising=False)
        assert env_int("RUNE_TEST_INT") is None

    def test_valid_positive(self, monkeypatch):
        monkeypatch.setenv("RUNE_TEST_INT", "120000")
        assert env_int("RUNE_TEST_INT", 1) == 120000

    def test_non_numeric_falls_back(self, monkeypatch):
        monkeypatch.setenv("RUNE_TEST_INT", "abc")
        assert env_int("RUNE_TEST_INT", 7) == 7
        assert env_int("RUNE_TEST_INT") is None

    @pytest.mark.parametrize("value", ["0", "-5"])
    def test_non_positive_falls_back(self, monkeypatch, value):
        # Non-positive values are treated as "unset" — callers use these for
        # caps/limits where <=0 is meaningless.
        monkeypatch.setenv("RUNE_TEST_INT", value)
        assert env_int("RUNE_TEST_INT", 9) == 9
        assert env_int("RUNE_TEST_INT") is None
