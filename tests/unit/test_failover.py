"""Tests for the failover module."""

from __future__ import annotations

import pytest

from rune.agent.failover import (
    FailoverManager,
    LLMProfile,
    classify_error,
    determine_strategy,
)


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
