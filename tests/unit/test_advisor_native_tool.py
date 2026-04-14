"""Unit tests for the Phase A Claude native advisor_20260301 tool path.

Covers:
- Eligibility matrix (provider + model prefix combinations)
- NativeAdvisorConfig resolution
- ToolWrapper construction with function=None passthrough
- Schema passthrough detection
- Synthetic event extraction from usage.iterations[]
- prompts.build_system_prompt advisor_native_enabled toggle
"""

from __future__ import annotations

from types import SimpleNamespace

from rune.agent.advisor.native_tool import (
    NativeAdvisorConfig,
    build_native_tool_wrapper,
    extract_synthetic_events_from_usage,
    is_native_schema,
    resolve_native_config,
)
from rune.agent.advisor.tiers import is_claude_native_eligible
from rune.agent.prompts import PROMPT_ADVISOR_TIMING, build_system_prompt


class TestEligibilityMatrix:
    def test_haiku_opus_eligible(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-haiku-4-5-20251001",
            "anthropic", "claude-opus-4-6",
        ) is True

    def test_sonnet_opus_eligible(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-sonnet-4-6",
            "anthropic", "claude-opus-4-6",
        ) is True

    def test_opus_opus_eligible(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-opus-4-6",
            "anthropic", "claude-opus-4-6",
        ) is True

    def test_cross_provider_executor_rejected(self):
        assert is_claude_native_eligible(
            "openai", "gpt-4o-mini",
            "anthropic", "claude-opus-4-6",
        ) is False

    def test_cross_provider_advisor_rejected(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-haiku-4-5",
            "openai", "o1",
        ) is False

    def test_non_canonical_advisor_rejected(self):
        # Sonnet as advisor isn't supported per Anthropic docs
        assert is_claude_native_eligible(
            "anthropic", "claude-sonnet-4-6",
            "anthropic", "claude-sonnet-4-6",
        ) is False

    def test_haiku_as_advisor_rejected(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-sonnet-4-6",
            "anthropic", "claude-haiku-4-5-20251001",
        ) is False

    def test_unrelated_model_executor_rejected(self):
        assert is_claude_native_eligible(
            "anthropic", "claude-3-opus-20240229",  # legacy, not in matrix
            "anthropic", "claude-opus-4-6",
        ) is False


class TestResolveNativeConfig:
    """``resolve_native_config`` requires both an Anthropic-eligible
    pair AND ``RUNE_ADVISOR_NATIVE=1`` to enable. Default is OFF
    because LiteLLM may not yet ship advisor_20260301 support."""

    def test_haiku_opus_pair_enabled(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config(
            "anthropic/claude-haiku-4-5-20251001",
            "anthropic/claude-opus-4-6",
        )
        assert cfg.enabled is True
        assert cfg.advisor_model == "claude-opus-4-6"
        assert cfg.beta_headers == {"anthropic-beta": "advisor-tool-2026-03-01"}

    def test_sonnet_opus_pair_enabled(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config(
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-opus-4-6",
        )
        assert cfg.enabled is True

    def test_default_off_without_env(self, monkeypatch):
        monkeypatch.delenv("RUNE_ADVISOR_NATIVE", raising=False)
        cfg = resolve_native_config(
            "anthropic/claude-haiku-4-5-20251001",
            "anthropic/claude-opus-4-6",
        )
        assert cfg.enabled is False  # Anthropic pair, but env not opted in

    def test_env_falsy_values_disable(self, monkeypatch):
        for falsy in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("RUNE_ADVISOR_NATIVE", falsy)
            cfg = resolve_native_config(
                "anthropic/claude-haiku-4-5-20251001",
                "anthropic/claude-opus-4-6",
            )
            assert cfg.enabled is False, f"falsy value {falsy!r} should disable"

    def test_openai_pair_disabled(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config(
            "openai/gpt-4o-mini",
            "openai/gpt-5.4",
        )
        assert cfg.enabled is False
        assert cfg.beta_headers == {}

    def test_no_advisor_disabled(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config("anthropic/claude-haiku-4-5", None)
        assert cfg.enabled is False
        assert cfg.beta_headers == {}

    def test_empty_advisor_disabled(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config("anthropic/claude-haiku-4-5", "")
        assert cfg.enabled is False

    def test_max_uses_propagates(self, monkeypatch):
        monkeypatch.setenv("RUNE_ADVISOR_NATIVE", "1")
        cfg = resolve_native_config(
            "anthropic/claude-haiku-4-5-20251001",
            "anthropic/claude-opus-4-6",
            max_uses=5,
        )
        assert cfg.max_uses == 5

    def test_beta_headers_disabled_returns_empty(self):
        cfg = NativeAdvisorConfig(enabled=False, advisor_model="")
        assert cfg.beta_headers == {}


class TestBuildNativeToolWrapper:
    def test_enabled_returns_wrapper(self):
        cfg = NativeAdvisorConfig(
            enabled=True, advisor_model="claude-opus-4-6", max_uses=3,
        )
        wrapper = build_native_tool_wrapper(cfg)
        assert wrapper is not None
        assert wrapper.name == "advisor"
        assert wrapper.function is None  # server-side, no client dispatch
        assert wrapper.json_schema["type"] == "advisor_20260301"
        assert wrapper.json_schema["name"] == "advisor"
        assert wrapper.json_schema["model"] == "claude-opus-4-6"
        assert wrapper.json_schema["max_uses"] == 3

    def test_disabled_returns_none(self):
        cfg = NativeAdvisorConfig(enabled=False, advisor_model="")
        assert build_native_tool_wrapper(cfg) is None

    def test_function_is_none_for_server_side(self):
        cfg = NativeAdvisorConfig(
            enabled=True, advisor_model="claude-opus-4-6",
        )
        wrapper = build_native_tool_wrapper(cfg)
        assert wrapper is not None
        # Server-side tool: must have no client function so the
        # client-side dispatch loop never tries to execute it.
        assert wrapper.function is None


class TestSchemaPassthrough:
    def test_advisor_20260301_detected(self):
        assert is_native_schema(
            {"type": "advisor_20260301", "name": "advisor"}
        ) is True

    def test_future_advisor_version_also_detected(self):
        # The prefix match means future versions (advisor_20270101 etc.)
        # automatically pass through without code changes.
        assert is_native_schema(
            {"type": "advisor_20270101", "name": "advisor"}
        ) is True

    def test_function_schema_not_detected(self):
        assert is_native_schema({"type": "function"}) is False

    def test_none_not_detected(self):
        assert is_native_schema(None) is False

    def test_non_dict_not_detected(self):
        assert is_native_schema("advisor_20260301") is False
        assert is_native_schema([]) is False

    def test_missing_type_not_detected(self):
        assert is_native_schema({"name": "advisor"}) is False


class TestExtractSyntheticEvents:
    def test_dict_iterations_with_advisor_message(self):
        usage = {
            "iterations": [
                {"type": "message", "output_tokens": 100},
                {
                    "type": "advisor_message",
                    "model": "claude-opus-4-6",
                    "output_tokens": 350,
                },
                {"type": "message", "output_tokens": 200},
            ]
        }
        events = extract_synthetic_events_from_usage(usage)
        assert len(events) == 1
        assert events[0]["trigger"] == "native"
        assert events[0]["model"] == "claude-opus-4-6"
        assert events[0]["output_tokens"] == 350
        assert events[0]["plan_injected"] is True
        assert events[0]["provider"] == "anthropic"

    def test_object_iterations_with_advisor_message(self):
        usage = SimpleNamespace(
            iterations=[
                SimpleNamespace(type="message", output_tokens=80),
                SimpleNamespace(
                    type="advisor_message",
                    model="claude-opus-4-6",
                    output_tokens=420,
                ),
            ]
        )
        events = extract_synthetic_events_from_usage(usage)
        assert len(events) == 1
        assert events[0]["output_tokens"] == 420

    def test_multiple_advisor_messages(self):
        usage = {
            "iterations": [
                {"type": "advisor_message", "model": "claude-opus-4-6",
                 "output_tokens": 300},
                {"type": "message", "output_tokens": 100},
                {"type": "advisor_message", "model": "claude-opus-4-6",
                 "output_tokens": 250},
            ]
        }
        events = extract_synthetic_events_from_usage(usage)
        assert len(events) == 2
        assert sum(e["output_tokens"] for e in events) == 550

    def test_no_advisor_messages_returns_empty(self):
        usage = {"iterations": [{"type": "message", "output_tokens": 100}]}
        assert extract_synthetic_events_from_usage(usage) == []

    def test_no_iterations_returns_empty(self):
        usage = {"input_tokens": 500, "output_tokens": 100}
        assert extract_synthetic_events_from_usage(usage) == []

    def test_none_usage_returns_empty(self):
        assert extract_synthetic_events_from_usage(None) == []

    def test_malformed_usage_gracefully_fails(self):
        # Object that raises on attribute access — should not crash
        class _Bad:
            @property
            def iterations(self):
                raise RuntimeError("kaboom")

        events = extract_synthetic_events_from_usage(_Bad())
        assert events == []

    def test_fallback_advisor_model_used_when_missing(self):
        usage = {"iterations": [{"type": "advisor_message", "output_tokens": 100}]}
        events = extract_synthetic_events_from_usage(
            usage, fallback_advisor_model="claude-opus-4-6",
        )
        assert events[0]["model"] == "claude-opus-4-6"


class TestSystemPromptInjection:
    def test_advisor_section_included_when_native_enabled(self):
        prompt = build_system_prompt(
            goal="test task",
            advisor_native_enabled=True,
        )
        assert "Advisor Tool Usage" in prompt
        assert "under 100 words" in prompt
        # Whitespace-flexible match for the durable directive
        normalized = " ".join(prompt.split())
        assert "make your deliverable durable" in normalized
        assert "BEFORE substantive work" in normalized

    def test_advisor_section_excluded_by_default(self):
        prompt = build_system_prompt(goal="test task")
        assert "Advisor Tool Usage" not in prompt

    def test_advisor_section_excluded_when_explicitly_false(self):
        prompt = build_system_prompt(
            goal="test task",
            advisor_native_enabled=False,
        )
        assert "Advisor Tool Usage" not in prompt

    def test_prompt_advisor_timing_constant_is_string(self):
        # Sanity: the constant exists and is non-trivial.
        assert isinstance(PROMPT_ADVISOR_TIMING, str)
        assert len(PROMPT_ADVISOR_TIMING) > 200
        assert "advisor" in PROMPT_ADVISOR_TIMING.lower()
