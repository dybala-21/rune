"""Advisor request/decision types. Read-only: state snapshot in, plan out.
Tool calls in advisor responses are stripped by normalizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AdvisorTrigger = Literal["early", "pre_done", "stuck", "reconcile", "native"]

AdvisorAction = Literal[
    "continue",
    "retry_tool",
    "switch_approach",
    "abort",
    "need_reconcile",
    "apply_patch",  # architect mode: advisor provides full file content
]


@dataclass(frozen=True, slots=True)
class AdvisorRequest:
    """Structured snapshot sent to the advisor at an escalation point."""

    trigger: AdvisorTrigger
    goal: str
    classification_summary: str
    activity_phase: str
    step: int
    token_budget_frac: float
    evidence_snapshot: dict[str, int]
    gate_state: dict[str, Any] | None
    stall_state: dict[str, int]
    recent_messages: list[dict[str, Any]]
    files_written: list[str]
    last_advisor_note: str | None = None
    # Architect mode only. Keyed by absolute path, values are raw file text.
    file_contents: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdvisorDecision:
    """Parsed advisor response.

    ``action`` is the first-line verb; ``plan_steps`` are the numbered
    guidance steps that get injected as a system message. ``raw_text``
    is preserved for audit/logging. When the advisor call fails or is
    disabled, the service returns ``AdvisorDecision.noop(reason)``.
    """

    action: AdvisorAction
    plan_steps: list[str]
    raw_text: str
    target_tool: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    provider: str = ""
    model: str = ""
    error_code: str | None = None
    trigger: AdvisorTrigger | None = None
    # Architect mode (action == "apply_patch"):
    #   patch = full file content to write
    #   patch_target_file = absolute path
    patch: str | None = None
    patch_target_file: str | None = None

    @staticmethod
    def noop(error_code: str, trigger: AdvisorTrigger | None = None) -> AdvisorDecision:
        return AdvisorDecision(
            action="continue",
            plan_steps=[],
            raw_text="",
            error_code=error_code,
            trigger=trigger,
        )


@dataclass(slots=True)
class AdvisorBudget:
    """Per-episode advisor budget tracking. Separate from executor token
    budget so advisor tokens never starve the main loop.

    ``call_history`` captures one structured dict per advisor call so the
    loop's episode-end write path can batch them into the advisor_events
    table (Tier 2). Contains metadata only — no plan text is retained."""

    max_calls: int = 3
    calls_used: int = 0
    tokens_used: int = 0
    disabled_reason: str | None = None
    call_history: list[dict[str, Any]] = field(default_factory=list)

    def can_call(self) -> bool:
        if self.disabled_reason is not None:
            return False
        if self.calls_used >= self.max_calls:
            return False
        return True

    def record(self, decision: AdvisorDecision) -> None:
        self.calls_used += 1
        self.tokens_used += max(0, decision.output_tokens)
        # plan_injected = the advisor returned a non-empty, non-error plan
        # that the loop will inject into the executor message stream.
        # It measures "guidance was applied" — NOT "executor actually
        # followed it at the behavioral level".
        plan_injected = bool(
            decision.error_code is None and decision.plan_steps
        )
        self.call_history.append(
            {
                "trigger": decision.trigger or "",
                "action": decision.action,
                "provider": decision.provider,
                "model": decision.model,
                "output_tokens": int(decision.output_tokens),
                "latency_ms": int(decision.latency_ms),
                "plan_injected": plan_injected,
                "error_code": decision.error_code or "",
            }
        )
