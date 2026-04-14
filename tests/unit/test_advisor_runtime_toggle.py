"""Unit tests for the runtime advisor on/off toggle."""

from __future__ import annotations

import pytest

from rune.agent.advisor import runtime_toggle
from rune.agent.advisor.runtime_toggle import (
    is_advisor_enabled,
    set_advisor_enabled,
)
from rune.agent.advisor.service import AdvisorConfig


@pytest.fixture
def isolated_toggle(tmp_path, monkeypatch):
    """Redirect the toggle file to a tmp path and wipe any env override.

    ``rune.utils.paths._HOME`` is frozen at import time, so monkeypatching
    HOME doesn't actually redirect ``~/.rune/data``. We instead patch
    ``_setting_path`` directly so every test is guaranteed hermetic.
    """
    monkeypatch.delenv("RUNE_ADVISOR_ENABLED", raising=False)
    toggle_file = tmp_path / "advisor_enabled"
    monkeypatch.setattr(
        runtime_toggle, "_setting_path", lambda: str(toggle_file),
    )
    return toggle_file


class TestIsAdvisorEnabled:
    def test_default_is_false_when_unset(self, isolated_toggle):
        # Fresh install: no env var, no toggle file → advisor is OFF.
        # Users must opt in explicitly via /advisor on or the web UI.
        assert is_advisor_enabled() is False

    def test_env_on_wins(self, isolated_toggle, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "on")
        assert is_advisor_enabled() is True

    def test_env_off_wins(self, isolated_toggle, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "off")
        assert is_advisor_enabled() is False

    def test_env_truthy_variants(self, isolated_toggle, monkeypatch):
        for val in ("1", "true", "yes", "ON"):
            monkeypatch.setenv("RUNE_ADVISOR_ENABLED", val)
            assert is_advisor_enabled() is True

    def test_env_falsy_variants(self, isolated_toggle, monkeypatch):
        for val in ("0", "false", "no", "OFF"):
            monkeypatch.setenv("RUNE_ADVISOR_ENABLED", val)
            assert is_advisor_enabled() is False

    def test_env_overrides_file(self, isolated_toggle, monkeypatch):
        set_advisor_enabled(False)
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "on")
        assert is_advisor_enabled() is True

    def test_file_off_persists(self, isolated_toggle):
        set_advisor_enabled(False)
        assert is_advisor_enabled() is False

    def test_file_on_persists(self, isolated_toggle):
        set_advisor_enabled(False)
        set_advisor_enabled(True)
        assert is_advisor_enabled() is True

    def test_unknown_env_value_falls_through_to_file(
        self, isolated_toggle, monkeypatch,
    ):
        set_advisor_enabled(False)
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "maybe")
        assert is_advisor_enabled() is False


class TestAdvisorConfigRespectsToggle:
    """from_env must short-circuit to disabled when toggle is off, even
    if RUNE_ADVISOR_MODEL is a valid pairing."""

    def test_toggle_off_forces_disabled(self, isolated_toggle, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        set_advisor_enabled(False)
        cfg = AdvisorConfig.from_env("anthropic/claude-haiku-4-5-20251001")
        assert cfg.enabled is False

    def test_toggle_on_allows_valid_pair(self, isolated_toggle, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        set_advisor_enabled(True)
        cfg = AdvisorConfig.from_env("anthropic/claude-haiku-4-5-20251001")
        assert cfg.enabled is True

    def test_toggle_default_blocks_even_with_model_set(
        self, isolated_toggle, monkeypatch,
    ):
        # Regression guard for Option A: even when RUNE_ADVISOR_MODEL
        # is set, a fresh install (no toggle file) must keep advisor
        # OFF so the web UI and actual behavior agree.
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        cfg = AdvisorConfig.from_env("anthropic/claude-haiku-4-5-20251001")
        assert cfg.enabled is False

    def test_env_override_off_forces_disabled(
        self, isolated_toggle, monkeypatch,
    ):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "off")
        cfg = AdvisorConfig.from_env("anthropic/claude-haiku-4-5-20251001")
        assert cfg.enabled is False
