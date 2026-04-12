"""Bridge between AgentLoop and AdvisorService.

build_policy_input is cheap (primitives only); build_advisor_request
deferred until needed. All exceptions become noop decisions.
"""

from __future__ import annotations

from typing import Any

from rune.agent.advisor.parser import format_injection
from rune.utils.logger import get_logger

log = get_logger(__name__)
from rune.agent.advisor.policy import (
    PolicyInput,
    mark_called,
    should_call,
)
from rune.agent.advisor.protocol import AdvisorDecision, AdvisorRequest
from rune.agent.advisor.service import AdvisorService


def build_policy_input(
    *,
    classification: Any,
    activity_phase: str,
    reads: int,
    writes: int,
    web_fetches: int,
    files_written: int,
    gate_blocked_count: int,
    stall_consecutive: int,
    no_progress_steps: int,
    wind_down_phase: str,
    hard_failures: int,
) -> PolicyInput:
    """Pure allocation of a ``PolicyInput`` from int/str primitives. No
    list copies, no I/O. Safe to call every step."""
    goal_type = getattr(classification, "goal_type", "") or ""
    is_complex = bool(getattr(classification, "is_complex_coding", False))
    return PolicyInput(
        is_complex_coding=is_complex,
        goal_type=str(goal_type),
        activity_phase=activity_phase,
        reads=reads,
        writes=writes,
        web_fetches=web_fetches,
        files_written=files_written,
        gate_blocked_count=gate_blocked_count,
        stall_consecutive=stall_consecutive,
        no_progress_steps=no_progress_steps,
        wind_down_phase=wind_down_phase,
        hard_failures=hard_failures,
    )


def build_advisor_request(
    *,
    trigger: str,
    goal: str,
    classification: Any,
    activity_phase: str,
    step: int,
    token_budget_frac: float,
    evidence: Any,
    gate_result: Any,
    stall_consecutive: int,
    stall_cumulative: int,
    recent_messages: list[Any],
    files_written: set[str] | list[str],
    hard_failures: list[str],
    last_advisor_note: str | None = None,
) -> AdvisorRequest:
    """Full snapshot construction — only called after should_call==True.

    This is the heavier helper: it copies message tails and file lists.
    Kept separate from ``build_policy_input`` so the hot path pays only
    for primitives.
    """
    classification_summary = (
        f"{getattr(classification, 'goal_type', '')} "
        f"complex={getattr(classification, 'is_complex_coding', False)}"
    )
    gate_state: dict[str, Any] | None = None
    if gate_result is not None:
        gate_state = {
            "outcome": getattr(gate_result, "outcome", "?"),
            "missing_requirement_ids": list(
                getattr(gate_result, "missing_requirement_ids", []) or []
            ),
            "hard_failures": list(hard_failures)[-5:],
        }
    msgs: list[dict[str, Any]] = []
    for m in recent_messages[-5:]:
        if isinstance(m, dict):
            msgs.append({"role": m.get("role", "?"), "content": m.get("content", "")})
    return AdvisorRequest(
        trigger=trigger,  # type: ignore[arg-type]
        goal=goal,
        classification_summary=classification_summary,
        activity_phase=activity_phase,
        step=step,
        token_budget_frac=token_budget_frac,
        evidence_snapshot={
            "reads": int(getattr(evidence, "reads", 0)),
            "writes": int(getattr(evidence, "writes", 0)),
            "executions": int(getattr(evidence, "executions", 0)),
            "web_searches": int(getattr(evidence, "web_searches", 0)),
            "web_fetches": int(getattr(evidence, "web_fetches", 0)),
        },
        gate_state=gate_state,
        stall_state={
            "consecutive": stall_consecutive,
            "cumulative": stall_cumulative,
        },
        recent_messages=msgs,
        files_written=list(files_written),
        last_advisor_note=last_advisor_note,
    )


async def maybe_consult(
    service: AdvisorService,
    *,
    policy_input: PolicyInput,
    build_request: Any,
    messages: list[Any],
    inject: Any,
) -> tuple[list[Any], AdvisorDecision | None]:
    """Orchestrate a single hook site.

    Parameters
    ----------
    service:
        The episode-scoped advisor service. Inert when disabled.
    policy_input:
        Already-built ``PolicyInput``. Caller reuses this cheap struct.
    build_request:
        Zero-arg callable returning an ``AdvisorRequest``. Deferred so
        message-tail / files copies only happen when ``should_call``
        returns True.
    messages:
        Current executor message list. Mutated through ``inject``.
    inject:
        Callable ``(messages, text) -> new_messages`` — usually
        ``AgentLoop._inject_system_message``.

    Returns
    -------
    (updated_messages, decision)
        ``decision`` is ``None`` when the service is disabled or the
        policy declined. When non-None, the decision is always the one
        returned by the service (possibly a ``noop`` on failure).
    """
    if not service.enabled:
        return messages, None
    call, trigger = should_call(service.policy_state, policy_input)
    if not call or trigger is None:
        return messages, None
    try:
        request = build_request(trigger)
    except Exception as exc:
        log.debug("advisor_request_build_failed", trigger=trigger, error=str(exc)[:200])
        return messages, None
    try:
        decision = await service.consult(request)
    except Exception as exc:
        log.debug("advisor_consult_failed", trigger=trigger, error=str(exc)[:200])
        return messages, None
    hard_failures = policy_input.hard_failures
    mark_called(
        service.policy_state,
        trigger,
        hard_failures=hard_failures,
        wind_down_phase=policy_input.wind_down_phase,
    )
    if decision.error_code is None and decision.plan_steps:
        service.policy_state.followed_last_advice = True
    # 2b-3: augment the freshly recorded call_history entry with the
    # stuck sub-reason. Policy set state.last_stuck_reason as a side
    # effect of should_call; budget.record fired inside service.consult
    # above so the last entry is the one to decorate.
    if trigger == "stuck" and service.budget.call_history:
        service.budget.call_history[-1]["stuck_reason"] = (
            service.policy_state.last_stuck_reason
        )
    injection_text = format_injection(decision)
    if injection_text:
        messages = inject(messages, injection_text)
    return messages, decision
