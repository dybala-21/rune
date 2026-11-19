"""An answer to a request about a schedule is backed by a scheduling tool that ran, or says it isn't."""

import json

import pytest
from pydantic import BaseModel

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import AgentConfig, CapabilityResult
from tests.unit.test_task_blocked import _astream, _delta


class Change(BaseModel):
    job_id: str
    enabled: bool = True


@pytest.mark.parametrize("case", ["called", "reminded", "never", "tool_failed", "not_about_a_schedule",
                                  "in_the_background"])
async def test_saying_it_is_stopped_needs_the_tool_that_stops_it(monkeypatch, tmp_path, case):
    for name, value in {
        "RUNE_HOME": str(tmp_path / "home"), "RUNE_IN_BEST_OF": "1", "RUNE_AUTO_SKILL": "0", "RUNE_AUTO_VERIFY": "0",
        "RUNE_REQUIREMENT_GATE": "0", "RUNE_ADVISOR": "0", "RUNE_BENCH_EVIDENCE_GATE": "0",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("RUNE_TOOL_ROUNDS", raising=False)
    from rune.config import get_config

    get_config().llm.route_simple_queries = False
    requests, changes = [], []
    registry = CapabilityRegistry()

    async def change(params):
        changes.append(params)
        if case == "tool_failed":
            return CapabilityResult(success=False, error="No such job")
        return CapabilityResult(success=True, output="The job is stopped.")

    registry.register(CapabilityDefinition(name="cron_update", description="Change or stop a scheduled job",
                                           group="schedule", parameters_model=Change, execute=change))

    async def completion(**kwargs):
        requests.append(kwargs)
        step = len(requests)
        assert step <= 4
        offered = [tool["function"]["name"] for tool in kwargs.get("tools") or []]
        reminded = "No scheduling tool has run in this turn" in str(kwargs["messages"])
        calls = {"called": step == 1, "reminded": step == 2, "tool_failed": step in (1, 3)}.get(case, False)
        if calls:
            assert "cron_update" in offered and reminded == (step > 1)
            return _astream([_delta(tool_calls=[{"index": 0, "id": f"call{step}", "name": "cron_update",
                                                 "arguments": json.dumps({"job_id": "b1", "enabled": False})}],
                                    finish_reason="tool_calls")])
        return _astream([_delta(content="Done. The briefing is stopped.", finish_reason="stop")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda: registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    classification = ClassificationResult(goal_type="chat", confidence=.99, tier=2,
                                          schedules=case != "not_about_a_schedule")
    if case == "in_the_background":
        # A scheduled run's own goal ("every morning, ...") reads as being about a schedule too.
        from rune.agent import background

        routine = background.BackgroundTask(goal="Every morning, list today's meetings", source="cron")
        monkeypatch.setattr(background, "current_background_task", lambda: routine)
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=6))
    loop._auto_skill = False
    loop._token_budget.total = 300_000
    trace = await loop._execute_loop(
        goal="Stop the morning briefing", system_prompt="Respond helpfully.",
        tools=loop._select_tools(classification), max_iterations=6, classification=classification,
        context={"workspace_root": str(tmp_path), "ephemeral": True},
    )
    from rune.api.trust import build_trust_payload

    payload = build_trust_payload(trace)
    backed = case in {"called", "reminded", "not_about_a_schedule", "in_the_background"}
    assert trace.reason == ("completed" if backed else "completed_gate_warnings")
    # called: tool, answer. reminded: answer, tool, answer. never: answer, answer.
    # tool_failed: tool, answer, tool again once told, answer.
    assert len(requests) == {"called": 2, "reminded": 3, "never": 2, "tool_failed": 4}.get(case, 1)
    if case in {"not_about_a_schedule", "in_the_background"}:
        assert "cron_update" not in [tool["function"]["name"] for tool in requests[0].get("tools") or []]
    assert len(changes) == {"called": 1, "reminded": 1, "tool_failed": 2}.get(case, 0)
    if backed:
        assert payload["completionCheck"] is None
    else:
        # The answer still reads "stopped"; beside it, what code knows: nothing went through.
        assert payload["completionCheck"] == {
            "name": "Schedule", "detail": "No scheduling tool ran in this turn, so nothing was set, changed or stopped."}
        assert payload["honestNote"] and not payload["verified"]
