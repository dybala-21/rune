"""Integration tests for the AdvisorService ↔ AgentLoop wiring.

Three scenarios:

1. ``maybe_consult`` fires on EARLY trigger after orientation reads.
2. ``maybe_consult`` fires on STUCK trigger at gate_blocked_count == 3
   and resets the counter when a plan is returned.
3. Advisor is fully inert when the service is disabled (env unset) —
   no calls to litellm, no state mutation.

These exercise the ``loop_integration`` helper together with a real
``AdvisorService`` instance backed by a mocked ``litellm.acompletion``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rune.agent.advisor.loop_integration import (
    build_advisor_request,
    build_policy_input,
    maybe_consult,
)
from rune.agent.advisor.service import AdvisorConfig, AdvisorService


@pytest.fixture(autouse=True)
def _force_advisor_toggle_on(monkeypatch):
    """Pin the runtime toggle ON so consult()'s live-recheck path does
    not short-circuit tests that construct an enabled AdvisorService
    directly. The toggle default is OFF on fresh install."""
    monkeypatch.setenv("RUNE_ADVISOR_ENABLED", "on")


def _mock_response(text: str):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=80)
    return SimpleNamespace(choices=[choice], usage=usage)


def _classification(goal_type: str = "code_modify", complex: bool = True):
    return SimpleNamespace(goal_type=goal_type, is_complex_coding=complex)


def _evidence(**overrides):
    base = dict(
        reads=0, writes=0, executions=0,
        web_searches=0, web_fetches=0, browser_reads=0,
        file_reads=0, unique_file_reads=0, verifications=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _enabled_service() -> AdvisorService:
    cfg = AdvisorConfig(
        enabled=True,
        provider="anthropic",
        model="claude-opus-4-6",
        timeout_ms=5_000,
        max_calls=3,
    )
    return AdvisorService(cfg)


def _inject(messages, text):
    return [*messages, {"role": "system", "content": text}]


def _build_request_closure(
    *,
    goal: str,
    classification,
    evidence,
    activity_phase: str,
    step: int,
    messages,
    files_written,
    hard_failures,
    gate_result=None,
):
    def _builder(trigger: str):
        return build_advisor_request(
            trigger=trigger,
            goal=goal,
            classification=classification,
            activity_phase=activity_phase,
            step=step,
            token_budget_frac=0.3,
            evidence=evidence,
            gate_result=gate_result,
            stall_consecutive=0,
            stall_cumulative=0,
            recent_messages=messages,
            files_written=files_written,
            hard_failures=hard_failures,
        )
    return _builder


@pytest.mark.asyncio
async def test_early_trigger_fires_after_orientation_reads():
    """After 2 file reads and zero writes, top-of-iteration hook fires
    EARLY and marks the policy state so it doesn't fire again."""
    service = _enabled_service()
    classification = _classification()
    evidence = _evidence(reads=2)
    messages = [{"role": "user", "content": "refactor helpers"}]

    pinput = build_policy_input(
        classification=classification,
        activity_phase="exploration",
        reads=evidence.reads,
        writes=evidence.writes,
        web_fetches=evidence.web_fetches,
        files_written=0,
        gate_blocked_count=0,
        stall_consecutive=0,
        no_progress_steps=0,
        wind_down_phase="normal",
        hard_failures=0,
    )

    mock = AsyncMock(return_value=_mock_response(
        "NEXT: continue\n1. read helpers.py\n2. outline structure"
    ))
    with patch("litellm.acompletion", new=mock):
        new_messages, decision = await maybe_consult(
            service,
            policy_input=pinput,
            build_request=_build_request_closure(
                goal="refactor helpers",
                classification=classification,
                evidence=evidence,
                activity_phase="exploration",
                step=3,
                messages=messages,
                files_written=set(),
                hard_failures=[],
            ),
            messages=messages,
            inject=_inject,
        )

    assert decision is not None
    assert decision.trigger == "early"
    assert decision.action == "continue"
    assert len(decision.plan_steps) == 2
    assert service.policy_state.early_called is True
    # Injection was appended to messages
    assert len(new_messages) == len(messages) + 1
    assert new_messages[-1]["role"] == "system"
    assert mock.await_count == 1

    # Second call is a no-op — early already fired
    with patch("litellm.acompletion", new=mock):
        _, decision2 = await maybe_consult(
            service,
            policy_input=pinput,
            build_request=_build_request_closure(
                goal="refactor helpers",
                classification=classification,
                evidence=evidence,
                activity_phase="exploration",
                step=4,
                messages=messages,
                files_written=set(),
                hard_failures=[],
            ),
            messages=new_messages,
            inject=_inject,
        )
    assert decision2 is None
    assert mock.await_count == 1  # unchanged


@pytest.mark.asyncio
async def test_stuck_trigger_fires_at_gate_blocked_3():
    """When gate_blocked_count hits 3, policy fires STUCK and the loop
    resets counters after the advisor returns a plan."""
    service = _enabled_service()
    classification = _classification()
    evidence = _evidence(reads=3, writes=1)
    messages = [{"role": "user", "content": "fix failing test"}]
    gate_result = SimpleNamespace(
        outcome="blocked",
        missing_requirement_ids=["R7", "R12"],
    )

    pinput = build_policy_input(
        classification=classification,
        activity_phase="verification",
        reads=evidence.reads,
        writes=evidence.writes,
        web_fetches=0,
        files_written=1,
        gate_blocked_count=3,
        stall_consecutive=0,
        no_progress_steps=0,
        wind_down_phase="normal",
        hard_failures=0,
    )

    mock = AsyncMock(return_value=_mock_response(
        "NEXT: retry_tool:bash_execute\n1. run pytest with -vv"
    ))
    with patch("litellm.acompletion", new=mock):
        new_messages, decision = await maybe_consult(
            service,
            policy_input=pinput,
            build_request=_build_request_closure(
                goal="fix failing test",
                classification=classification,
                evidence=evidence,
                activity_phase="verification",
                step=9,
                messages=messages,
                files_written={"test_x.py"},
                hard_failures=[],
                gate_result=gate_result,
            ),
            messages=messages,
            inject=_inject,
        )

    assert decision is not None
    assert decision.trigger == "stuck"
    assert decision.action == "retry_tool"
    assert decision.target_tool == "bash_execute"
    assert service.policy_state.stuck_calls == 1
    assert service.policy_state.followed_last_advice is True
    assert len(new_messages) == len(messages) + 1


@pytest.mark.asyncio
async def test_disabled_service_is_fully_inert(monkeypatch):
    """When RUNE_ADVISOR_MODEL is unset, from_env returns disabled
    config, maybe_consult never invokes litellm, and no state
    changes. This is the regression guard for the default install."""
    monkeypatch.delenv("RUNE_ADVISOR_MODEL", raising=False)
    cfg = AdvisorConfig.from_env("claude-sonnet-4-5-20250929")
    assert cfg.enabled is False
    service = AdvisorService(cfg)
    assert service.enabled is False

    classification = _classification()
    evidence = _evidence(reads=5, writes=3)
    messages = [{"role": "user", "content": "anything"}]
    pinput = build_policy_input(
        classification=classification,
        activity_phase="verification",
        reads=5, writes=3, web_fetches=0, files_written=2,
        gate_blocked_count=3,
        stall_consecutive=10,
        no_progress_steps=5,
        wind_down_phase="final",
        hard_failures=2,
    )

    mock = AsyncMock()
    with patch("litellm.acompletion", new=mock):
        new_messages, decision = await maybe_consult(
            service,
            policy_input=pinput,
            build_request=_build_request_closure(
                goal="anything",
                classification=classification,
                evidence=evidence,
                activity_phase="verification",
                step=20,
                messages=messages,
                files_written={"a.py", "b.py"},
                hard_failures=["err1", "err2"],
            ),
            messages=messages,
            inject=_inject,
        )

    assert decision is None
    assert new_messages is messages  # unchanged
    assert mock.await_count == 0  # litellm never called
    assert service.budget.calls_used == 0
    assert service.policy_state.early_called is False
    assert service.policy_state.stuck_calls == 0
