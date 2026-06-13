"""Tests for the failover module."""

from __future__ import annotations

import pytest

from rune.agent.failover import (
    FailoverManager,
    LLMProfile,
    build_profiles_from_config,
    classify_error,
    determine_strategy,
)


class TestBuildProfilesFromConfig:
    """The primary profile is what the agent loop runs, so it must use the
    session selection even when only one half (provider or model) was
    overridden. An all-or-nothing check would upgrade a partial override to the
    default provider's best tier.
    """

    def _set_llm(self, monkeypatch, provider=None, model=None, default="openai"):
        from rune.config import get_config

        cfg = get_config()
        monkeypatch.setattr(cfg.llm, "default_provider", default)
        monkeypatch.setattr(cfg.llm, "active_provider", provider)
        monkeypatch.setattr(cfg.llm, "active_model", model)
        return cfg

    def test_model_only_active_keeps_user_model(self, monkeypatch):
        # `rune -m gpt-5-mini` (no -p) must NOT upgrade to openai's best tier.
        self._set_llm(monkeypatch, provider=None, model="gpt-5-mini")
        primary = build_profiles_from_config()[0]
        assert (primary.provider, primary.model) == ("openai", "gpt-5-mini")

    def test_provider_only_active_uses_that_providers_best(self, monkeypatch):
        cfg = self._set_llm(monkeypatch, provider="anthropic", model=None)
        primary = build_profiles_from_config()[0]
        assert primary.provider == "anthropic"
        assert primary.model == cfg.llm.models.anthropic.best

    def test_both_active_win(self, monkeypatch):
        self._set_llm(
            monkeypatch, provider="anthropic", model="claude-haiku-4-5-20251001"
        )
        primary = build_profiles_from_config()[0]
        assert (primary.provider, primary.model) == (
            "anthropic",
            "claude-haiku-4-5-20251001",
        )

    def test_no_active_falls_back_to_default_best(self, monkeypatch):
        cfg = self._set_llm(monkeypatch)
        primary = build_profiles_from_config()[0]
        assert primary.provider == "openai"
        assert primary.model == cfg.llm.models.openai.best


class TestClassifyError:
    def test_classify_error_auth(self):
        assert classify_error("401 unauthorized") == "auth"
        assert classify_error(Exception("invalid api key")) == "auth"

    def test_classify_error_rate_limit(self):
        assert classify_error("429 rate limit exceeded") == "rate_limit"
        assert classify_error("too many requests") == "rate_limit"

    def test_classify_error_timeout(self):
        assert classify_error("request timed out") == "timeout"
        assert classify_error("deadline exceeded") == "timeout"

    def test_classify_error_context(self):
        assert classify_error("context_length_exceeded") == "context_overflow"
        assert classify_error("maximum context length") == "context_overflow"

    def test_classify_error_unknown(self):
        assert classify_error("something weird happened") == "unknown"


class TestDetermineStrategy:
    def _make_profile(self, name: str = "test", priority: int = 0, thinking: str = "none") -> LLMProfile:
        return LLMProfile(
            name=name, provider="openai", model="gpt-4", priority=priority,
            thinking_level=thinking,  # type: ignore[arg-type]
        )

    def test_determine_strategy_rate_limit_retry(self):
        profile = self._make_profile()
        profiles = [profile, self._make_profile("alt", priority=1)]
        strategy = determine_strategy("rate_limit", profile, retries_left=2, profiles=profiles)
        assert strategy.action == "retry"
        assert strategy.delay > 0

    def test_determine_strategy_auth_switch(self):
        profile = self._make_profile("main", priority=0)
        alt = self._make_profile("alt", priority=1)
        strategy = determine_strategy("auth", profile, retries_left=3, profiles=[profile, alt])
        assert strategy.action == "switch_profile"
        assert strategy.new_profile is not None
        assert strategy.new_profile.name == "alt"

    def test_determine_strategy_context_compact(self):
        profile = self._make_profile(thinking="none")
        strategy = determine_strategy(
            "context_overflow", profile, retries_left=1, profiles=[profile],
        )
        assert strategy.action == "compact"


class TestFailoverManager:
    @pytest.mark.asyncio
    async def test_failover_manager_handle_error(self):
        profiles = [
            LLMProfile(name="a", provider="p", model="m", priority=0),
            LLMProfile(name="b", provider="p", model="m", priority=1),
        ]
        mgr = FailoverManager(profiles=profiles, max_retries=2)

        result = await mgr.handle_error("rate limit exceeded")
        assert result.success is True
        assert result.reason == "rate_limit"

    @pytest.mark.asyncio
    async def test_failover_manager_profile_switching(self):
        profiles = [
            LLMProfile(name="a", provider="p", model="m", priority=0),
            LLMProfile(name="b", provider="p", model="m", priority=1),
        ]
        mgr = FailoverManager(profiles=profiles, max_retries=1)
        assert mgr.current_profile.name == "a"

        result = await mgr.handle_error("unauthorized")
        assert result.success is True
        assert mgr.current_profile.name == "b"

    @pytest.mark.asyncio
    async def test_failover_manager_reset(self):
        profiles = [
            LLMProfile(name="a", provider="p", model="m", priority=0),
            LLMProfile(name="b", provider="p", model="m", priority=1),
        ]
        mgr = FailoverManager(profiles=profiles, max_retries=3)
        # Switch profile via auth error
        await mgr.handle_error("unauthorized")
        assert mgr.current_profile.name == "b"

        mgr.reset()
        assert mgr.current_profile.name == "a"
        assert mgr.retries_left == 3

    @pytest.mark.asyncio
    async def test_reduce_thinking_level(self):
        profiles = [
            LLMProfile(
                name="thinker", provider="p", model="m",
                thinking_level="extended", priority=0,
            ),
        ]
        mgr = FailoverManager(profiles=profiles, max_retries=3)

        # extended -> basic on context overflow
        result = await mgr.handle_error("context_length_exceeded")
        assert result.success is True
        assert mgr.current_profile.thinking_level == "basic"

        # basic -> compact (not reduce_thinking since basic is not "extended")
        result = await mgr.handle_error("context_length_exceeded")
        assert result.success is True
