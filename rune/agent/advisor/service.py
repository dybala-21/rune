"""Provider-agnostic advisor orchestration. Resolves model, validates
pairing, builds payload, invokes LiteLLM, parses response, tracks budget.
Singleton per episode via ``AdvisorService.for_episode``."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from rune.agent.advisor.context_fitter import build_payload
from rune.agent.advisor.normalizer import normalize
from rune.agent.advisor.parser import parse
from rune.agent.advisor.policy import PolicyState
from rune.agent.advisor.protocol import (
    AdvisorBudget,
    AdvisorDecision,
    AdvisorRequest,
)
from rune.agent.advisor.tiers import (
    MIN_TIER_GAP,
    check_pairing,
    extract_provider_and_model,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)


_ADVISOR_SYSTEM_PROMPT = """You are an ADVISOR. You will NOT take actions. You will NOT call tools.

Respond in under 100 words, enumerated. Start with EXACTLY one verb line:
NEXT: continue | retry_tool:<name> | switch_approach | abort | need_reconcile

Then 1-5 numbered steps, no prose.

If the executor's evidence contradicts a prior advisor note, prefer the
evidence. Be concrete — name tools, file paths, and missing requirements.
"""

# Per-provider soft timeout (ms). Reasoning models get longer budgets.
# Gemini 2.5-pro observed burning 300~700 reasoning tokens per advisor call
# (measured in P0-2 bench), routinely exceeding the original 15s budget on
# the 'reconcile' trigger. Bumped to 60s to match openai/deepseek reasoning.
_TIMEOUT_MS: dict[str, int] = {
    "anthropic":  15_000,
    "openai":     60_000,
    "deepseek":   60_000,
    "gemini":     60_000,
    "xai":        15_000,
    "ollama":    120_000,
    "":           30_000,
}

_MAX_CONSECUTIVE_FAILURES = 3
_DEFAULT_MAX_CALLS = 3


@dataclass(slots=True)
class AdvisorConfig:
    """Runtime configuration resolved at episode start."""

    enabled: bool
    provider: str
    model: str
    timeout_ms: int
    max_calls: int = _DEFAULT_MAX_CALLS

    @staticmethod
    def from_env(executor_model: str) -> AdvisorConfig:
        """Read RUNE_ADVISOR_MODEL + runtime toggle. Returns disabled on
        unset, invalid pairing, or toggle off."""
        from rune.agent.advisor.runtime_toggle import is_advisor_enabled
        if not is_advisor_enabled():
            return AdvisorConfig(
                enabled=False, provider="", model="", timeout_ms=0,
            )
        raw = os.environ.get("RUNE_ADVISOR_MODEL", "").strip()
        if not raw:
            return AdvisorConfig(
                enabled=False, provider="", model="", timeout_ms=0,
            )
        adv_provider, adv_model = extract_provider_and_model(raw)
        exec_provider, exec_model_name = extract_provider_and_model(executor_model)
        pairing = check_pairing(
            executor_provider=exec_provider,
            executor_model=exec_model_name,
            advisor_provider=adv_provider,
            advisor_model=adv_model,
            min_gap=MIN_TIER_GAP,
        )
        if not pairing.ok:
            log.warning(
                "advisor_pairing_invalid",
                executor=executor_model,
                advisor=raw,
                reason=pairing.reason,
            )
            return AdvisorConfig(
                enabled=False, provider="", model="", timeout_ms=0,
            )
        timeout = _TIMEOUT_MS.get(adv_provider, _TIMEOUT_MS[""])
        return AdvisorConfig(
            enabled=True,
            provider=adv_provider,
            model=adv_model,
            timeout_ms=timeout,
        )


class AdvisorService:
    """One instance per episode. Thread/async-safe for sequential loop
    use (the loop awaits each consult before proceeding)."""

    def __init__(self, config: AdvisorConfig) -> None:
        self._config = config
        self.budget = AdvisorBudget(max_calls=config.max_calls)
        self.policy_state = PolicyState(max_stuck_calls=max(1, config.max_calls - 1))
        self._consecutive_failures = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self.budget.disabled_reason is None

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def model_full(self) -> str:
        """Return provider-qualified advisor model id.

        Used by ``loop.py`` to build the native advisor config so it
        can decide whether to inject the ``advisor_20260301`` tool.
        Empty string when the advisor service is disabled.
        """
        if not self._config.enabled:
            return ""
        return f"{self._config.provider}/{self._config.model}"

    @staticmethod
    def for_episode(executor_model: str) -> AdvisorService:
        cfg = AdvisorConfig.from_env(executor_model)
        return AdvisorService(cfg)

    # Main entry point

    async def consult(
        self,
        request: AdvisorRequest,
    ) -> AdvisorDecision:
        """Consult the advisor. Never raises; returns ``noop`` on any
        failure. Enforces per-episode budget."""
        if not self._config.enabled:
            return AdvisorDecision.noop("disabled", trigger=request.trigger)
        # Live toggle recheck: honor mid-episode /advisor off from TUI or
        # web without waiting for the next episode to pick up the change.
        from rune.agent.advisor.runtime_toggle import is_advisor_enabled
        if not is_advisor_enabled():
            return AdvisorDecision.noop("toggled_off", trigger=request.trigger)
        if not self.budget.can_call():
            return AdvisorDecision.noop(
                self.budget.disabled_reason or "budget_exhausted",
                trigger=request.trigger,
            )

        t0 = time.monotonic()
        try:
            raw = await self._invoke_llm(request)
        except TimeoutError:
            self._record_failure()
            decision = AdvisorDecision.noop("timeout", trigger=request.trigger)
            self.budget.record(decision)
            log.warning(
                "advisor_timeout",
                trigger=request.trigger,
                provider=self._config.provider,
            )
            return decision
        except Exception as exc:
            self._record_failure()
            decision = AdvisorDecision.noop("error", trigger=request.trigger)
            self.budget.record(decision)
            log.warning(
                "advisor_invoke_failed",
                trigger=request.trigger,
                provider=self._config.provider,
                error=str(exc)[:200],
            )
            return decision

        latency_ms = int((time.monotonic() - t0) * 1000)
        text = normalize(raw)

        usage = _extract_usage(raw)
        decision = parse(
            text,
            trigger=request.trigger,
            provider=self._config.provider,
            model=self._config.model,
            input_tokens=usage[0],
            output_tokens=usage[1],
            latency_ms=latency_ms,
        )
        if decision.error_code is None:
            self._consecutive_failures = 0
        else:
            self._record_failure()
        self.budget.record(decision)
        log.info(
            "advisor_consulted",
            trigger=request.trigger,
            action=decision.action,
            steps=len(decision.plan_steps),
            latency_ms=latency_ms,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            provider=self._config.provider,
            model=self._config.model,
        )
        return decision

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self.budget.disabled_reason = "consecutive_failures"
            self.policy_state.advisor_disabled = True
            log.warning(
                "advisor_disabled",
                reason="consecutive_failures",
                count=self._consecutive_failures,
            )

    async def _invoke_llm(self, request: AdvisorRequest) -> Any:
        """Single-shot LiteLLM call. Non-streaming, no tools, bounded
        output, per-provider timeout. One implementation for every
        provider LiteLLM supports."""
        import litellm

        payload = build_payload(request)
        from rune.agent.litellm_adapter import _resolve_litellm_model

        raw_model = f"{self._config.provider}/{self._config.model}"
        resolved_model, extra = _resolve_litellm_model(raw_model)

        messages = [
            {"role": "system", "content": _ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]

        timeout_s = max(1.0, self._config.timeout_ms / 1000.0)
        # 4096 keeps room for reasoning tokens (o1, gemini-2.5-pro, etc.)
        # while still bounding runaway output. Advisor text itself stays
        # small (100 words) per the system prompt.
        return await asyncio.wait_for(
            litellm.acompletion(
                model=resolved_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4_096,
                stream=False,
                **extra,
            ),
            timeout=timeout_s,
        )


def _extract_usage(raw: Any) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` from any LiteLLM response
    shape. Returns ``(0, 0)`` if unavailable — we treat missing usage
    conservatively rather than guessing."""
    try:
        usage = getattr(raw, "usage", None)
        if usage is None and isinstance(raw, dict):
            usage = raw.get("usage")
        if usage is None:
            return (0, 0)
        if isinstance(usage, dict):
            return (
                int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            )
        prompt = int(
            getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        completion = int(
            getattr(usage, "completion_tokens", 0)
            or getattr(usage, "output_tokens", 0)
            or 0
        )
        return (prompt, completion)
    except Exception:
        return (0, 0)
