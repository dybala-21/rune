"""Source checks through the streaming adapter, including a mistaken chat route."""

import json

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.capabilities.web import WebFetchParams, WebSearchParams
from rune.types import AgentConfig, CapabilityResult
from tests.unit.test_task_blocked import _astream, _delta


@pytest.mark.parametrize("case", ["chat", "missing", "memory_only", "search", "fetch", "web_lookup", "empty", "failed", "recovered", "misrouted", "reserve", "reserve_empty", "explicit_limit", "continued_lookup", "explicit_research", "current_repaired", "current_refused", "current_excerpt"])
async def test_sources_are_required_without_redundant_model_calls(monkeypatch, tmp_path, case):
    for name, value in {
        "RUNE_HOME": str(tmp_path / "home"), "RUNE_IN_BEST_OF": "1",
        "RUNE_AUTO_SKILL": "0", "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0",
        "RUNE_ADVISOR": "0", "RUNE_BENCH_EVIDENCE_GATE": "0",
        "RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT": "0", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(name, value)
    requests, actions = [], []
    monkeypatch.delenv("RUNE_TOOL_ROUNDS", raising=False)
    from rune.config import get_config

    get_config().llm.route_simple_queries = False
    if case.startswith("reserve") or case == "web_lookup":
        monkeypatch.setattr("rune.agent.loop._compute_tool_rounds", lambda *args, **kwargs: 1)
    if case == "explicit_limit":
        monkeypatch.setenv("RUNE_TOOL_ROUNDS", "3")
    registry = CapabilityRegistry()

    async def retrieve(params):
        actions.append(params)
        if case == "current_excerpt" and isinstance(params, WebFetchParams):
            return CapabilityResult(success=True, output="Quotes unavailable; 20 minute delay")
        if case in {"empty", "reserve_empty"}:
            return CapabilityResult(success=True, output="No search results found.", metadata={"count": 0})
        if case == "failed":
            return CapabilityResult(success=False, error="Source unavailable")
        return CapabilityResult(success=True, output="Observed value: 120 on 2026-10-15", metadata={"count": 1})

    registry.register(CapabilityDefinition(name="web_search", description="Search",
                                          parameters_model=WebSearchParams, execute=retrieve))
    registry.register(CapabilityDefinition(name="web_fetch", description="Read page",
                                          parameters_model=WebFetchParams, execute=retrieve))
    from rune.capabilities.memory_capability import MemorySearchParams

    async def recall(params):
        return CapabilityResult(success=True, output="An old preference, not a current source")

    registry.register(CapabilityDefinition(name="memory_search", description="Recall",
                                          parameters_model=MemorySearchParams, execute=recall))

    async def completion(**kwargs):
        requests.append(kwargs)
        step = len(requests)
        assert step <= 5
        if case == "current_excerpt" and step == 1:
            return _astream([_delta(tool_calls=[
                {"index": 0, "id": "search1", "name": "web_search", "arguments": json.dumps({"query": "current quote"})},
                {"index": 1, "id": "fetch1", "name": "web_fetch", "arguments": json.dumps({"url": "https://example.org/quote"})},
            ], finish_reason="tool_calls")])
        lookup = (step == 1 and case in {"search", "fetch", "web_lookup", "empty", "failed", "misrouted", "memory_only", "reserve", "reserve_empty"}
                  or step == 2 and case == "recovered"
                  or step == 1 and case in {"current_repaired", "current_refused"}
                  or step == 3 and case == "current_repaired")
        lookup = lookup or case == "explicit_limit" and step <= 3 or case in {"continued_lookup", "explicit_research"} and step <= 4
        if lookup:
            if case == "recovered" or case == "current_repaired" and step == 3:
                assert "[Source check]" in str(kwargs["messages"])
            name = "web_fetch" if case in {"fetch", "web_lookup"} or case == "current_repaired" and step == 3 else "memory_search" if case == "memory_only" else "web_search"
            args = {"url": "https://example.org/prices"} if name == "web_fetch" else {"query": f"current conditions {step}"}
            return _astream([_delta(tool_calls=[{"index": 0, "id": f"call{step}", "name": name,
                                                "arguments": json.dumps(args)}], finish_reason="tool_calls")])
        if case.startswith("reserve") or case in {"explicit_limit", "web_lookup"}:
            assert not kwargs.get("tools") and "No more tools" in str(kwargs["messages"])
        if case in {"continued_lookup", "explicit_research"}:
            assert kwargs.get("tools")
        return _astream([_delta(content="Current price: 120" if case == "current_excerpt" else "Here is my response.", finish_reason="stop")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda: registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    classification = ClassificationResult(goal_type="web" if case in {"explicit_research", "web_lookup"} else "chat", confidence=.99, tier=2,
                                         needs_web_evidence=case not in {"chat", "misrouted"},
                                         lookup_only=case in {"fetch", "web_lookup"},
                                         needs_current_source=case in {"current_repaired", "current_refused", "current_excerpt"})
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.6" if case in {"fetch", "web_lookup"} else "openai/gpt-5.4", max_iterations=4))
    loop._auto_skill = False
    loop._token_budget.total = 300_000
    trace = await loop._execute_loop(
        goal="Respond to the current concern", system_prompt="Respond helpfully.",
        tools=loop._select_tools(classification), max_iterations=4, classification=classification,
        context={"workspace_root": str(tmp_path), "ephemeral": True},
    )
    should_complete = case in {"chat", "search", "fetch", "web_lookup", "recovered", "misrouted", "reserve", "explicit_limit", "continued_lookup", "explicit_research", "current_repaired"}
    assert trace.reason == ("completed" if should_complete else "completed_gate_warnings")
    if not should_complete:
        assert trace.completion_check["name"] == "Source retrieval"
    assert len(actions) == (2 if case in {"current_repaired", "current_excerpt"} else 1 if case == "current_refused" else 3 if case == "explicit_limit" else 4 if case in {"continued_lookup", "explicit_research"} else
                            int(case in {"search", "fetch", "web_lookup", "empty", "failed", "recovered", "misrouted", "reserve", "reserve_empty"}))
    assert len(requests) == (4 if case in {"explicit_limit", "current_repaired"} else 5 if case in {"continued_lookup", "explicit_research"} else
                            1 if case == "chat" else 2 if case in {"missing", "search", "fetch", "web_lookup", "misrouted", "reserve", "reserve_empty"} else 3)
    assert all(call.get("stream") for call in requests)
    efforts = [call.get("reasoning_effort", call.get("extra_body", {}).get("reasoning", {}).get("effort")) for call in requests]
    assert efforts == [("low" if case in {"fetch", "web_lookup"} else None)] * len(requests)
