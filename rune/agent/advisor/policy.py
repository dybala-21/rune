"""Escalation policy — decides when the executor should consult the advisor.

Rules (all provider-agnostic):

1. EARLY — once orientation has happened (>=2 reads or >=1 web_fetch)
   and no writes yet, call the advisor once before substantive work.
2. PRE-DONE — immediately before a completion gate evaluation, if the
   deliverable is durable (>=1 write AND files_written non-empty).
3. STUCK — on gate_blocked==3, stall, no-progress threshold, or
   wind_down transition to final, call the advisor once per trigger.
4. RECONCILE — if the executor followed an advisor note and the next
   state got worse (new hard failure / still blocked), allow exactly one
   reconciliation call per episode.

Complex-coding gate: if the goal classification says this is trivial
chat/QA, the advisor is fully disabled — matches Anthropic's "don't use
for short reactive tasks" guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

from rune.agent.advisor.protocol import AdvisorTrigger


@dataclass(slots=True)
class PolicyState:
    """Mutable per-episode counters read by ``should_call``. Updated by
    the loop at key signal moments or by the service after each call."""

    early_called: bool = False
    pre_done_called: bool = False
    stuck_calls: int = 0
    reconcile_called: bool = False
    max_stuck_calls: int = 2
    followed_last_advice: bool = False
    advisor_disabled: bool = False
    hard_failures_at_last_advice: int = 0
    # P2: Rising-edge tracking for wind_down_phase. wind_down is
    # monotonic forward (none → wrapping → stopping → final → hard_stop)
    # because token_budget.used only increases during a run, so once the
    # phase hits "final" it stays there until hard_stop. Without this
    # guard, the stuck trigger would re-fire on every iteration while
    # in "final" — wasting up to 2 advisor calls on the same signal.
    last_wind_down_phase: str = "normal"
    # 2b-3: last stuck sub-condition that fired, for observability /
    # self-improving feedback. Populated as a side effect of
    # ``should_call`` when it returns a stuck trigger so the loop can
    # persist it to the advisor_events row. Empty string when the last
    # trigger was not stuck (or no trigger fired yet).
    last_stuck_reason: str = ""


@dataclass(frozen=True, slots=True)
class PolicyInput:
    is_complex_coding: bool
    goal_type: str
    activity_phase: str
    reads: int
    writes: int
    web_fetches: int
    files_written: int
    gate_blocked_count: int
    stall_consecutive: int
    no_progress_steps: int
    wind_down_phase: str
    hard_failures: int


def _is_trivial_goal(goal_type: str, is_complex_coding: bool) -> bool:
    if is_complex_coding:
        return False
    # Only "chat" is a valid trivial GoalType per rune.agent.goal_classifier;
    # other goal_types (research, web, code_modify, ...) go through full
    # trigger evaluation.
    if goal_type == "chat":
        return True
    return False


def should_call(
    state: PolicyState,
    inp: PolicyInput,
) -> tuple[bool, AdvisorTrigger | None]:
    """Return (should_call, trigger) for the current signal snapshot.

    Only fires on rising-edge transitions: the caller is expected to
    invoke this after state updates and rely on the internal counters
    to avoid double-firing.
    """
    if state.advisor_disabled:
        return False, None
    if _is_trivial_goal(inp.goal_type, inp.is_complex_coding):
        return False, None

    # 4. RECONCILE (highest priority — only when eligible)
    if (
        not state.reconcile_called
        and state.followed_last_advice
        and inp.hard_failures > state.hard_failures_at_last_advice
    ):
        return True, "reconcile"

    # 3. STUCK
    # Sub-reason is recorded on state.last_stuck_reason as a side
    # effect so the loop's persist path can attribute each stuck call
    # to the specific signal that triggered it (enables measuring P2's
    # rising-edge effect + future per-reason policy tuning).
    if state.stuck_calls < state.max_stuck_calls:
        if inp.gate_blocked_count == 3:
            state.last_stuck_reason = "gate_blocked"
            return True, "stuck"
        if inp.stall_consecutive >= 5:
            state.last_stuck_reason = "stall"
            return True, "stuck"
        if inp.no_progress_steps >= 3:
            state.last_stuck_reason = "no_progress"
            return True, "stuck"
        # Rising-edge only: fire exactly once when transitioning into
        # "final". Re-fires on the same persistent state are blocked via
        # state.last_wind_down_phase (updated by mark_called).
        if (
            inp.wind_down_phase == "final"
            and state.last_wind_down_phase != "final"
        ):
            state.last_stuck_reason = "wind_down"
            return True, "stuck"

    # 2. PRE-DONE
    if (
        not state.pre_done_called
        and inp.writes >= 1
        and inp.files_written >= 1
        and inp.activity_phase in ("verification", "implementation")
    ):
        return True, "pre_done"

    # 1. EARLY
    if (
        not state.early_called
        and inp.writes == 0
        and (inp.reads >= 2 or inp.web_fetches >= 1)
    ):
        return True, "early"

    return False, None


def mark_called(
    state: PolicyState,
    trigger: AdvisorTrigger,
    *,
    hard_failures: int,
    wind_down_phase: str = "normal",
) -> None:
    """Record that an advisor call happened for the given trigger so
    the same rising-edge doesn't re-fire.

    ``wind_down_phase`` is the current phase at the time of the call;
    it is stored so the next ``should_call`` can detect whether the
    state is still the same persistent signal (and block re-fire) or
    a fresh transition (and allow fire). Default preserves backward
    compatibility for callers that don't track wind-down.
    """
    state.hard_failures_at_last_advice = hard_failures
    state.last_wind_down_phase = wind_down_phase
    if trigger == "early":
        state.early_called = True
    elif trigger == "pre_done":
        state.pre_done_called = True
    elif trigger == "stuck":
        state.stuck_calls += 1
    elif trigger == "reconcile":
        state.reconcile_called = True
