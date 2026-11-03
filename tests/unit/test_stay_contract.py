"""Request-bound completion and the lifetime of the evidence actually returned."""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rune.agent.browser_requirements import request_key
from rune.agent.classification_response import InvalidClassification, validate_decision
from rune.agent.goal_classifier import ClassificationResult, classify_goal, from_wire, to_wire
from rune.agent.loop import NativeAgentLoop
from rune.agent.retrieval_result import accepted_answer
from rune.agent.run_outcome import run_outcome
from rune.agent.stay_contract import bound_query
from rune.agent.task_evidence import TaskEvidence
from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
from rune.agent.tool_errors import ToolPreparationError
from rune.capabilities.browser.stay_evidence import BrowserStayEvidenceParams
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult
from tests.unit.test_goal_classifier import client, response, verdict
from tests.unit.test_stay_evidence import TEXT, bound_evidence, capture, complete, fields, url
from tests.unit.test_stay_evidence import ledger as ledger

GOAL = "Compare three distinct Sapporo hotels for 2026-11-15 to 2026-11-16, 2 adults, under 200000 KRW."


def classification(ledger, goal=GOAL, count=3):
    query = {**ledger.query.model_dump(mode="json"), "count": count}
    return ClassificationResult("browser", .99, 2, browser_effect="view",
                               stay_query=query, stay_query_key=request_key((goal,)))


@pytest.mark.parametrize("route", ["browser", "web"])
@pytest.mark.parametrize("goal", [
    GOAL,
    "NOL에서 2026-11-15부터 2026-11-16까지 성인 2명, 20만원 이하 삿포로 숙소 3곳 비교해줘.",
])
async def test_request_conditions_share_the_existing_routing_call(monkeypatch, ledger, goal, route):
    query = classification(ledger, goal).stay_query
    stub = client(monkeypatch, response(verdict(goal_type=route, browser_effect="view", stay_query=query)))
    routed = await classify_goal(goal)
    assert routed.available and routed.stay_query == query
    assert routed.stay_query_key == request_key((goal,))
    assert stub.completion.await_count == 1
    sent = stub.completion.call_args.kwargs
    schema = sent["response_format"]["json_schema"]["schema"]
    assert "stay_query" in schema["required"]
    query_schema = schema["properties"]["stay_query"]["anyOf"][0]
    assert set(query_schema["required"]) == set(query)
    assert "Never reduce the requested count" in sent["messages"][0]["content"]
    assert from_wire(to_wire(routed)) == routed
    evidence = TaskEvidence.for_request(goal, routed)
    assert bound_query(evidence).count == 3
    evidence.prepare_tool("browser_stay_evidence", {"action": "begin", "query": query})


@pytest.mark.parametrize("change", [
    {"count": 0}, {"count": 11}, {"check_out": "2026-11-14"}, {"adults": 0}, {"rooms": 2},
])
def test_invalid_router_contract_cannot_enter_execution(ledger, change):
    query = {**classification(ledger).stay_query, **change}
    with pytest.raises(InvalidClassification, match="stay_query"):
        validate_decision(verdict(goal_type="browser", browser_effect="view", stay_query=query))


def test_router_must_not_default_an_omitted_requested_count(ledger):
    query = classification(ledger).stay_query
    del query["count"]
    with pytest.raises(InvalidClassification, match="stay_query"):
        validate_decision(verdict(goal_type="browser", browser_effect="view", stay_query=query))


@pytest.mark.parametrize("change", [
    {"count": 1}, {"city": "Tokyo"}, {"check_in": "2026-11-14"},
    {"check_out": "2026-11-17"}, {"adults": 1}, {"max_total_krw": 300000},
])
async def test_changed_collector_contract_is_refused_before_dispatch(ledger, change):
    evidence = TaskEvidence.for_request(GOAL, classification(ledger))
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    cap = CapabilityDefinition(name="browser_stay_evidence", description="collect",
                               parameters_model=BrowserStayEvidenceParams, execute=execute)
    registry = CapabilityRegistry()
    registry.register(cap)
    tool = _build_typed_tool(cap_def=cap, reg=registry, cache=None, stall=None,
                             opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool))
    query = {**classification(ledger).stay_query, **change}
    result = await tool.function(action="begin", query=query)
    assert "stay_request_mismatch" in result
    execute.assert_not_awaited()


@pytest.mark.parametrize("binding", ["missing", "another_request", "new_user_instruction"])
def test_unbound_or_reused_contract_cannot_complete_or_start(ledger, binding):
    complete(ledger)
    evidence = bound_evidence(ledger, GOAL)
    if binding == "missing":
        evidence.stay_query = None
    elif binding == "another_request":
        evidence.stay_query_key = request_key(("A different request",))
    else:
        evidence.request_texts += ("Actually compare three Tokyo hotels",)
    evidence.retrieval_acceptance = ledger.acceptance()
    assert not accepted_answer(evidence)
    with pytest.raises(ToolPreparationError, match="stay_request_unbound"):
        evidence.prepare_tool("browser_stay_evidence", {"action": "begin", "query": ledger.query.model_dump(mode="json")})


def test_collector_metadata_cannot_rebind_the_request(ledger):
    complete(ledger)
    evidence = TaskEvidence.for_request(GOAL, classification(ledger))
    expected = dict(evidence.stay_query)
    evidence.observe("browser_stay_evidence", CapabilityResult(
        success=True, metadata={"retrieval_acceptance": ledger.acceptance(), "stay_query": ledger.query.model_dump(mode="json")}))
    assert evidence.stay_query == expected
    assert not accepted_answer(evidence)


@pytest.mark.parametrize("change", [
    {"city": "Tokyo"}, {"check_in": "2026-11-14"}, {"check_out": "2026-11-17"},
    {"adults": 1}, {"max_total_krw": 300000},
])
def test_final_acceptance_checks_conditions_even_when_property_counts_match(ledger, change):
    complete(ledger)
    evidence = bound_evidence(ledger)
    evidence.stay_query.update(change)
    evidence.retrieval_acceptance = ledger.acceptance()
    assert len(ledger.result()["offers"]) == evidence.stay_query["count"]
    assert not accepted_answer(evidence)


def test_unused_expired_source_does_not_expire_fresh_selected_evidence(ledger):
    unused = capture(ledger, url("TDP-99"), TEXT + "Unused earlier observation\n", "old")
    unused.observed_at = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
    used = complete(ledger)
    report = ledger.acceptance()
    assert report["status"] == "pass"
    assert report["expires_at"] == (datetime.fromisoformat(used.observed_at) + timedelta(minutes=15)).isoformat()
    evidence = bound_evidence(ledger)
    evidence.retrieval_acceptance = report
    assert json.loads(accepted_answer(evidence)) == ledger.result()
    # The unused source is kept for audit/readback; deletion is not the fix.
    assert unused.id in ledger.sources


def test_expired_selected_evidence_still_blocks_completion(ledger):
    used = complete(ledger)
    used.observed_at = (datetime.now(UTC) - timedelta(minutes=16)).isoformat()
    capture(ledger, url("TDP-99"), TEXT + "Fresh but unused observation\n", "new")
    ledger.finished = True
    evidence = bound_evidence(ledger)
    evidence.retrieval_acceptance = ledger.acceptance()
    assert not accepted_answer(evidence)
    assert ledger.result()["limitations"]


@pytest.mark.parametrize("collected,bound,stop,success", [
    (1, True, "normal", False), (1, True, "budget", False),
    (1, False, "normal", False), (1, False, "budget", False),
    (3, True, "normal", True), (3, True, "budget", True),
])
async def test_loop_completes_only_the_original_requested_count(monkeypatch, tmp_path, ledger, collected, bound, stop, success):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    ledger.query.count = collected
    for index in range(collected):
        source = capture(ledger, url(f"TDP-{index + 1}"), TEXT, f"doc-{index}")
        ledger.record(fields(source), 90000)
    ledger.finished = True
    options = []

    class Stream:
        tool_budget_exhausted = False
        tool_round_limit_reached = False
        answer_reserve_reason = ""

        async def stream_text(self, **kwargs):
            await options[-1].on_tool_start("browser_stay_evidence", {"action": "finish"})
            await options[-1].on_tool_end("browser_stay_evidence", CapabilityResult(
                success=True, output=json.dumps(ledger.result()),
                metadata={"retrieval_acceptance": ledger.acceptance()}))
            if stop == "budget":
                from rune.agent.request_budget import BudgetExceeded
                raise BudgetExceeded("No more model calls")
            yield "The collector finished."

        async def get_output(self):
            return "The collector finished."

        def usage(self):
            return SimpleNamespace(input_tokens=10, output_tokens=10)

        def get_failure_state(self):
            return {}, set()

        def all_messages(self):
            return []

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def native_advisor_events(self):
            return []

        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            yield Stream()

    def build(opts):
        options.append(opts)
        return {}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    routed = classification(ledger) if bound else ClassificationResult("browser", .99, 2, browser_effect="view")
    trace = await loop._execute_loop(GOAL, "", [], 1, routed)
    assert run_outcome(trace).success is success
    assert trace.requirement_acceptance["status"] == ("pass" if success else "inconclusive")
    if success:
        assert len(json.loads(loop._last_answer_text)["offers"]) == 3
    else:
        assert trace.reason != "completed"
        assert json.loads(loop._last_answer_text)["limitations"]
