"""Behavioral Compliance Tracker — observe whether the executor follows advice.

After advisor injects a plan, the loop tracks tool calls and evidence for
a grace period, then issues a verdict: followed or ignored. Counters
(stall, gate_blocked) are only reset on a positive verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.agent.advisor.protocol import AdvisorDecision


@dataclass(slots=True)
class PendingAdvice:
    """Tracks one injected advisor plan until verdict."""

    decision: AdvisorDecision
    injected_at_step: int
    grace_steps: int = 2

    expected_tool: str | None = None
    expected_action: str = ""

    baseline_evidence: int = 0
    baseline_hard_failures: int = 0

    observed_tools: list[str] = field(default_factory=list)

    # Index into AdvisorBudget.call_history for merging verdicts
    # into the persisted event record.
    call_history_index: int = -1
    advice_mode: str = ""


@dataclass(frozen=True, slots=True)
class ComplianceVerdict:
    followed: bool
    reason: str  # "tool_matched" | "progress_made" | "ignored" | "worse"
    evidence_delta: int
    failure_delta: int


def check_compliance(
    pending: PendingAdvice,
    current_step: int,
    current_evidence: int,
    current_hard_failures: int,
    recent_tool_calls: list[str],
) -> ComplianceVerdict | None:
    """Return None during grace period, verdict after."""
    pending.observed_tools.extend(recent_tool_calls)

    if current_step - pending.injected_at_step < pending.grace_steps:
        return None

    ev_delta = current_evidence - pending.baseline_evidence
    fail_delta = current_hard_failures - pending.baseline_hard_failures

    if pending.expected_tool and pending.expected_tool in pending.observed_tools:
        return ComplianceVerdict(True, "tool_matched", ev_delta, fail_delta)

    if ev_delta > 0 and fail_delta <= 0:
        return ComplianceVerdict(True, "progress_made", ev_delta, fail_delta)

    if fail_delta > 0:
        return ComplianceVerdict(False, "worse", ev_delta, fail_delta)

    return ComplianceVerdict(False, "ignored", ev_delta, fail_delta)


def build_pending(
    decision: AdvisorDecision,
    step: int,
    evidence_total: int,
    hard_failure_count: int,
    *,
    call_history_index: int = -1,
    advice_mode: str = "",
) -> PendingAdvice:
    """Create a PendingAdvice from an advisor decision.

    For apply_patch actions, the expected tool is file_write since the
    executor's only job is to mechanically apply the advisor's patch.
    """
    expected_tool = decision.target_tool
    if decision.action == "apply_patch" and decision.patch:
        expected_tool = "file_write"
    return PendingAdvice(
        decision=decision,
        injected_at_step=step,
        expected_tool=expected_tool,
        expected_action=decision.action,
        baseline_evidence=evidence_total,
        baseline_hard_failures=hard_failure_count,
        call_history_index=call_history_index,
        advice_mode=advice_mode,
    )
