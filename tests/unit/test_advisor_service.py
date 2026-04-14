"""Unit tests for AdvisorService with mocked LiteLLM backend.

The service is provider-agnostic because it calls ``litellm.acompletion``
directly. We mock that single function to test the full pipeline
(config → invoke → normalize → parse → budget) without any network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rune.agent.advisor.protocol import AdvisorRequest
from rune.agent.advisor.service import AdvisorConfig, AdvisorService


@pytest.fixture(autouse=True)
def _force_advisor_toggle_on(monkeypatch):
    """Pin the runtime toggle ON for every test in this module so a
    local ~/.rune/data/advisor_enabled=off file can't suppress the
    config under test. Individual tests that care about the toggle
    live in test_advisor_runtime_toggle.py."""
    monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "on")


def _request(trigger: str = "stuck") -> AdvisorRequest:
    return AdvisorRequest(
        trigger=trigger,  # type: ignore[arg-type]
        goal="fix a bug",
        classification_summary="code_modify complex",
        activity_phase="implementation",
        step=10,
        token_budget_frac=0.5,
        evidence_snapshot={"reads": 3, "writes": 1},
        gate_state={"outcome": "blocked", "missing_requirement_ids": ["R1"]},
        stall_state={"consecutive": 1},
        recent_messages=[{"role": "user", "content": "hi"}],
        files_written=["x.py"],
    )


def _litellm_response(text: str, prompt_tokens: int = 400, completion_tokens: int = 80):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class TestAdvisorConfigFromEnv:
    def test_unset_disables(self, monkeypatch):
        monkeypatch.delenv("RUNE_ADVISOR_MODEL", raising=False)
        cfg = AdvisorConfig.from_env("claude-sonnet-4-5-20250929")
        assert cfg.enabled is False

    def test_valid_pair_enables(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        cfg = AdvisorConfig.from_env("claude-sonnet-4-5-20250929")
        assert cfg.enabled is True
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-opus-4-6"

    def test_invalid_pair_disables(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-haiku-4-5-20251001")
        cfg = AdvisorConfig.from_env("claude-opus-4-6")
        assert cfg.enabled is False

    def test_hybrid_local_executor_cloud_advisor(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "anthropic/claude-opus-4-6")
        cfg = AdvisorConfig.from_env("ollama/qwen2.5:7b")
        assert cfg.enabled is True

    def test_pure_local(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_MODEL", "ollama/qwen2.5-qwq")
        cfg = AdvisorConfig.from_env("ollama/qwen2.5:7b")
        assert cfg.enabled is True
        assert cfg.provider == "ollama"


class TestAdvisorServiceConsult:
    @pytest.mark.asyncio
    async def test_disabled_returns_noop(self):
        cfg = AdvisorConfig(enabled=False, provider="", model="", timeout_ms=0)
        svc = AdvisorService(cfg)
        d = await svc.consult(_request())
        assert d.error_code == "disabled"
        assert d.action == "continue"

    @pytest.mark.asyncio
    async def test_live_toggle_off_mid_episode(self, monkeypatch):
        """A runtime /advisor off must short-circuit subsequent consult()
        calls even after the service was constructed with enabled=True."""
        cfg = AdvisorConfig(
            enabled=True,
            provider="anthropic",
            model="claude-opus-4-6",
            timeout_ms=5_000,
        )
        svc = AdvisorService(cfg)
        # Flip the toggle off after construction.
        monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "off")
        # litellm must NOT be invoked — if it is, the test explodes.
        boom = AsyncMock(side_effect=AssertionError("litellm should not be called"))
        with patch("litellm.acompletion", new=boom):
            d = await svc.consult(_request())
        assert d.error_code == "toggled_off"
        assert d.action == "continue"
        # Budget is untouched — the call never left the service.
        assert svc.budget.calls_used == 0

    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        cfg = AdvisorConfig(
            enabled=True,
            provider="anthropic",
            model="claude-opus-4-6",
            timeout_ms=5_000,
        )
        svc = AdvisorService(cfg)
        mock_response = _litellm_response(
            "NEXT: retry_tool:file_read\n1. re-read with larger offset"
        )
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=mock_response),
        ):
            d = await svc.consult(_request())
        assert d.error_code is None
        assert d.action == "retry_tool"
        assert d.target_tool == "file_read"
        assert d.input_tokens == 400
        assert d.output_tokens == 80
        assert svc.budget.calls_used == 1

    @pytest.mark.asyncio
    async def test_timeout_is_handled(self, monkeypatch):
        cfg = AdvisorConfig(
            enabled=True,
            provider="anthropic",
            model="claude-opus-4-6",
            timeout_ms=10,
        )
        svc = AdvisorService(cfg)

        async def _slow(*args, **kwargs):
            import asyncio
            await asyncio.sleep(5)
            return None

        with patch("litellm.acompletion", new=_slow):
            d = await svc.consult(_request())
        assert d.error_code == "timeout"
        assert d.action == "continue"
        assert svc.budget.calls_used == 1

    @pytest.mark.asyncio
    async def test_consecutive_failures_disable_advisor(self):
        cfg = AdvisorConfig(
            enabled=True,
            provider="anthropic",
            model="claude-opus-4-6",
            timeout_ms=5_000,
            max_calls=10,
        )
        svc = AdvisorService(cfg)
        boom = AsyncMock(side_effect=RuntimeError("network"))
        with patch("litellm.acompletion", new=boom):
            for _ in range(3):
                await svc.consult(_request())
        assert svc.budget.disabled_reason == "consecutive_failures"
        assert svc.policy_state.advisor_disabled is True

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_calls(self):
        cfg = AdvisorConfig(
            enabled=True,
            provider="anthropic",
            model="claude-opus-4-6",
            timeout_ms=5_000,
            max_calls=2,
        )
        svc = AdvisorService(cfg)
        ok = _litellm_response("NEXT: continue\n1. step")
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=ok),
        ):
            await svc.consult(_request())
            await svc.consult(_request())
            d = await svc.consult(_request())
        assert d.error_code is not None

    @pytest.mark.asyncio
    async def test_deepseek_r1_think_tags_are_stripped(self):
        cfg = AdvisorConfig(
            enabled=True,
            provider="deepseek",
            model="deepseek-reasoner",
            timeout_ms=5_000,
        )
        svc = AdvisorService(cfg)
        text = "<think>long reasoning</think>\nNEXT: abort\n1. give up"
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_litellm_response(text)),
        ):
            d = await svc.consult(_request())
        assert d.action == "abort"
        assert "<think>" not in d.raw_text or d.raw_text == ""
