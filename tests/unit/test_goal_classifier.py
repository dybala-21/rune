"""Routing decisions must be complete before tools can be selected."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rune.agent.classification_response import RESPONSE_FORMAT
from rune.agent.goal_classifier import classify_goal, from_wire, to_wire


def verdict(**changes):
    return {"goal_type": "full", "confidence": 0.9, "reason": "requested work",
            "requires_execution": False, "intent_categories": [],
            "needs_web_evidence": False, "needs_current_source": False,
            "lookup_only": False, "source_output": "none", "browser_effect": "change", "calculation_operation": "other",
            "requires_desktop_input": False, "is_related_to_previous": False,
            "table_output": "none", "calculation_expression": "", **changes}


def response(data, finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {
        "content": data if isinstance(data, str) else json.dumps(data),
    }}]}


def client(monkeypatch, *responses):
    stub = AsyncMock()
    stub.completion.side_effect = responses
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: stub)
    return stub


async def test_valid_routing_needs_one_constrained_call(monkeypatch):
    stub = client(monkeypatch, response(verdict(goal_type="code_modify", requires_execution=True, intent_categories=["coding"])))
    result = await classify_goal("Fix the test and summarize the result")
    assert result.available and result.requires_execution and result.is_complex_coding
    assert stub.completion.await_count == 1
    assert stub.completion.call_args.kwargs["response_format"] == RESPONSE_FORMAT
    assert from_wire(to_wire(result)) == result


async def test_browser_requirements_share_the_routing_call(monkeypatch):
    from rune.agent.browser_requirements import request_key
    goal = "Adults to 3 and Children to 3"
    requirements = {"complete": True, "items": [
        {"requirement": "Adults to 3", "expected": "3", "kind": "value"},
        {"requirement": "Children to 3", "expected": "3", "kind": "value"},
    ]}
    stub = client(monkeypatch, response(verdict(goal_type="browser", browser_requirements=requirements)))
    result = await classify_goal(goal)
    assert result.available and result.browser_requirements == requirements
    assert result.browser_requirements_key == request_key((goal,))
    assert stub.completion.await_count == 1
    assert from_wire(to_wire(result)) == result


async def test_resume_task_must_be_one_of_the_supplied_records(monkeypatch):
    stub = client(monkeypatch, response(verdict(resume_task="ffffffffffff")), response(verdict(resume_task="ffffffffffff")))
    result = await classify_goal("Continue that task", saved_tasks=[{"id":"aabbccddeeff", "request":"Update report"}])
    assert not result.available and stub.completion.await_count == 2
    assert "invalid_resume_task" in result.reason


async def test_file_roles_do_not_replace_the_saved_task_schema(monkeypatch):
    from rune.agent.classification_response import request_classification

    stub = client(monkeypatch, response(verdict(resume_task="aabbccddeeff", file_roles=[{"name":"report.csv", "role":"input"}])))
    roles = {}
    values = await request_classification(stub, "Route", json.dumps({
        "request_to_classify":"Continue editing report.csv", "saved_tasks":[{"id":"aabbccddeeff", "request":"Edit report.csv"}],
    }), file_roles=roles)
    properties = stub.completion.call_args.kwargs["response_format"]["json_schema"]["schema"]["properties"]
    assert "resume_task" in properties and "file_roles" in properties
    assert values["resume_task"] == "aabbccddeeff" and roles == {"report.csv":"input"}


@pytest.mark.parametrize("lookup_only", [True, False])
async def test_lookup_scope_reuses_the_existing_classification(monkeypatch, lookup_only):
    stub = client(monkeypatch, response(verdict(goal_type="chat", needs_web_evidence=True, lookup_only=lookup_only)))
    result = await classify_goal("Help with this current concern")
    assert result.available and result.lookup_only is lookup_only
    assert from_wire(to_wire(result)).lookup_only is lookup_only
    assert stub.completion.await_count == 1


@pytest.mark.parametrize("changes", [{"needs_web_evidence": False}, {"goal_type": "code_modify"},
                                     {"requires_execution": True}, {"intent_categories": ["document"]}])
def test_lookup_optimization_cannot_change_the_task_scope(changes):
    from rune.agent.classification_response import validate_decision

    values = validate_decision(verdict(goal_type="chat", needs_web_evidence=True, lookup_only=True) | changes)
    assert values["lookup_only"] is False
    assert all(values[key] == value for key, value in changes.items())


def test_current_source_cannot_bypass_the_source_requirement():
    from rune.agent.classification_response import validate_decision

    values = validate_decision(verdict(needs_current_source=True))
    assert values["needs_current_source"] and values["needs_web_evidence"]


@pytest.mark.parametrize("changes,expected", [
    ({}, True), ({"goal_type": "research"}, True),
    ({"complexity": "complex", "is_multi_task": True}, True),
    ({"goal_type": "browser"}, False), ({"goal_type": "artifact"}, False),
    ({"browser_effect": "edit"}, False), ({"browser_effect": "change"}, False),
    ({"requires_execution": True}, False), ({"requires_code": True}, False),
    ({"requires_desktop_input": True}, False), ({"is_complex_coding": True}, False),
    ({"output_expectation": "file"}, False), ({"output_expectation": "either"}, False),
    ({"intent_categories": frozenset({"email"})}, False),
    ({"needs_web_evidence": False}, False), ({"available": False}, False),
])
def test_research_answer_reserve_preserves_actions_and_deliverables(changes, expected):
    from rune.agent.goal_classifier import ClassificationResult, is_web_research

    classification = ClassificationResult(**{
        "goal_type": "web", "confidence": 0.9, "tier": 2,
        "needs_web_evidence": True, "browser_effect": "view", **changes,
    })
    assert is_web_research(classification) is expected
    assert not is_web_research(None)


@pytest.mark.parametrize("items", [
    [{"requirement": "Change people to three", "expected": "three"}],
    [{"requirement": "confirm it", "kind": "result", "confirms": "missing"}],
])
async def test_invalid_optional_browser_plan_defers_to_executor_without_rerouting(monkeypatch, items):
    stub = client(monkeypatch, response(verdict(goal_type="browser",
        browser_requirements={"complete": True, "items": items})))
    result = await classify_goal("Set people to 3, save and confirm it")
    assert result.available and result.goal_type == "browser"
    assert result.browser_requirements is None
    assert stub.completion.await_count == 1


@pytest.mark.parametrize("route,requirement", [("chat", "read"), ("artifact", "write"),
                                               ("code_modify", "write"), ("execution", "execute")])
async def test_source_requirement_preserves_the_requested_outcome(monkeypatch, route, requirement):
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.loop import NativeAgentLoop

    stub = client(monkeypatch, response(verdict(goal_type=route, needs_web_evidence=True)))
    result = await classify_goal("Complete the request using current sources")
    assert result.available and stub.completion.await_count == 1
    contract = resolve_intent_contract(result, result.confidence)
    assert contract.tool_requirement == requirement
    assert contract.grounding_requirement == "required"
    loop = NativeAgentLoop()
    loop._auto_skill = False
    assert {"web_search", "web_fetch", "task_blocked"} <= set(loop._select_tools(result))


@pytest.mark.parametrize("route,intents,software,write", [
    ("artifact", ["document", "table"], False, False),
    ("artifact", [], False, False),
    ("code_modify", ["document", "table"], False, False),
    ("full", ["document", "table"], False, False),
    ("full", ["desktop"], False, False),
    ("code_modify", ["coding"], True, True),
    ("full", ["coding", "document"], True, True),
    ("research", ["coding"], True, False),
])
async def test_requested_domain_controls_coding_guidance(monkeypatch, route, intents, software, write):
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.loop import NativeAgentLoop, _compute_explore_budget
    from rune.agent.prompts import PROMPT_CODE, PROMPT_COMPLEX_TASK

    stub = client(monkeypatch, response(verdict(goal_type=route, intent_categories=intents,
                                              table_output="xlsx" if "table" in intents else "none")))
    result = await classify_goal("Requested outcome")
    assert result.available and result.requires_code is write
    assert result.is_complex_coding is write
    assert (_compute_explore_budget(result) > 0) is write
    assert stub.completion.await_count == 1
    contract = resolve_intent_contract(result, 0.9)
    assert contract.requires_code_verification is write
    assert contract.requires_code_write_artifact is write
    if "document" in intents:
        assert contract.tool_requirement == "write"
        assert contract.output_expectation == "file"

    from unittest.mock import Mock
    repo_map = Mock(return_value="project symbols")
    monkeypatch.setattr("rune.intelligence.repo_map.build_repo_map_sync", repo_map)
    prompt = await NativeAgentLoop()._build_system_prompt("Requested outcome", result)
    assert (PROMPT_CODE in prompt) is software
    assert (PROMPT_COMPLEX_TASK in prompt) is write
    assert repo_map.call_count == int(software)
    assert from_wire(to_wire(result)) == result


@pytest.mark.parametrize("route,intents,software,write", [
    ("code_modify", [], True, True),
    ("code_modify", ["coding"], True, True),
    ("code_modify", ["document", "table"], False, False),
    ("artifact", [], False, False),
    ("research", ["coding"], True, False),
])
def test_restored_classification_keeps_coding_requirements(route, intents, software, write):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.prompts import PROMPT_CODE, build_system_prompt

    classification = ClassificationResult(route, 0.9, 2, requires_execution=True,
                                          intent_categories=frozenset(intents))
    restored = from_wire(to_wire(classification))
    contract = resolve_intent_contract(restored, 0.9)
    prompt = build_system_prompt("Requested outcome", classification=restored)
    assert (PROMPT_CODE in prompt) is software
    assert contract.requires_code_verification is write
    assert contract.requires_code_write_artifact is write
    assert (contract.kind == "code_write") is write


async def test_calculation_extraction_reuses_the_routing_call(monkeypatch):
    from rune.agent.calculation import calculation_context

    stub = client(monkeypatch, response(verdict(goal_type="chat", calculation_expression="0.1 + 0.2")))
    result = await classify_goal("Evaluate 0.1 + 0.2")
    assert '"result": "0.3"' in calculation_context("Evaluate 0.1 + 0.2", result)
    assert stub.completion.await_count == 1
    assert from_wire(to_wire(result)) == result
    schema = stub.completion.call_args.kwargs["response_format"]["json_schema"]["schema"]
    assert "verbatim" in schema["properties"]["calculation_expression"]["description"]


async def test_extra_prose_is_rejected_and_retried_once(monkeypatch):
    malformed = '```json\n' + json.dumps(verdict()) + '\n```\nI will now implement the request.'
    stub = client(monkeypatch, response(malformed), response(verdict()))
    result = await classify_goal("Fix amounts.py and show before/after tests")
    assert result.available and stub.completion.await_count == 2
    assert "model" in stub.completion.call_args.kwargs


@pytest.mark.parametrize("change", [
    {"intent_categories": None}, {"requires_execution": "false"},
    {"requires_desktop_input": "false"}, {"confidence": float("nan")},
    {"goal_type": "invented"}, {"intent_categories": ["unknown"]},
    {"goal_type": "artifact", "intent_categories": ["coding"]},
])
async def test_invalid_decision_cannot_enable_tools(monkeypatch, change):
    bad = response(verdict(**change))
    stub = client(monkeypatch, bad, bad)
    result = await classify_goal("Operate the selected app")
    assert not result.available and stub.completion.await_count == 2
    assert from_wire(to_wire(result)).available is False


async def test_missing_field_is_not_silently_defaulted(monkeypatch):
    data = verdict()
    del data["requires_desktop_input"]
    client(monkeypatch, response(data), response(data))
    assert not (await classify_goal("Use the app")).available


@pytest.mark.parametrize("finish", ["length", "max_tokens", "content_filter"])
async def test_truncated_or_filtered_response_is_not_accepted(monkeypatch, finish):
    client(monkeypatch, response(verdict(), finish), response(verdict(), finish))
    assert not (await classify_goal("Summarize this")).available


@pytest.mark.parametrize("output,expected", [("none", False), ("csv", True), ("xlsx", True)])
async def test_table_gate_uses_deliverable_not_response_layout(monkeypatch, output, expected):
    client(monkeypatch, response(verdict(intent_categories=["table"], table_output=output)))
    result = await classify_goal("Summarize the result")
    assert result.available and ("table" in result.intent_categories) is expected


@pytest.mark.parametrize("needs_input", [True, False])
async def test_desktop_intent_keeps_explicit_input_requirement(monkeypatch, needs_input):
    client(monkeypatch, response(verdict(intent_categories=["desktop"], requires_desktop_input=needs_input)))
    result = await classify_goal("Inspect the selected app")
    assert result.requires_desktop_input is needs_input
    assert not result.requires_execution


async def test_unrelated_followup_does_not_inherit_desktop(monkeypatch):
    client(monkeypatch, response(verdict(goal_type="chat")))
    result = await classify_goal("Summarize these notes", previous_goal="Use TextEdit", previous_goal_type="full")
    assert result.is_domain_change and not result.intent_categories


async def test_live_browser_context_reuses_the_existing_routing_call(monkeypatch):
    stub = client(monkeypatch, response(verdict(goal_type="web", is_related_to_previous=True)))
    state = {"status": "open", "url": "https://example.test/list", "title": "Results"}
    result = await classify_goal("Read the first result", previous_goal="Find the list",
                                 previous_goal_type="web", browser_state=state)
    assert result.available and not result.is_domain_change
    assert stub.completion.await_count == 1
    messages = stub.completion.call_args.kwargs["messages"]
    assert json.loads(messages[-1]["content"])["browser_state"] == state


async def test_provider_error_fails_closed_without_unbounded_retry(monkeypatch):
    stub = client(monkeypatch, RuntimeError("provider unavailable"))
    result = await classify_goal("Save a document")
    assert not result.available and stub.completion.await_count == 1
    assert "RuntimeError" in result.reason


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (400, 1), (429, 2), (503, 2)])
async def test_only_transient_rejections_retry_within_the_same_budget(monkeypatch, status, expected_calls):
    import httpx

    reply = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"),
                           headers={"Retry-After": "0"})
    error = httpx.HTTPStatusError("private provider details", request=reply.request, response=reply)
    stub = client(monkeypatch, error, response(verdict(goal_type="web")))
    result = await classify_goal("Find the source")
    assert stub.completion.await_count == expected_calls
    assert result.available == (expected_calls == 2)
    requests = [call.kwargs for call in stub.completion.call_args_list]
    assert all(call["max_retries"] == 0 for call in requests)
    assert 30 < requests[0]["timeout"] <= 35
    if expected_calls == 2:
        assert requests[1]["timeout"] <= requests[0]["timeout"]


async def test_provider_retry_after_cannot_extend_routing_deadline(monkeypatch):
    import httpx

    reply = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"),
                           headers={"Retry-After": "120"})
    stub = client(monkeypatch, httpx.HTTPStatusError("limited", request=reply.request, response=reply))
    result = await classify_goal("Read the page")
    assert not result.available and stub.completion.await_count == 1


@pytest.mark.parametrize("remaining", [35.0, 0.5])
async def test_request_timeout_respects_the_cap_and_inherited_deadline(monkeypatch, remaining):
    import asyncio
    from types import SimpleNamespace

    from rune.agent import classification_response as contract

    now = 250.1  # Adding and subtracting 35 at this clock value rounds upward.
    monkeypatch.setattr(contract, "asyncio", SimpleNamespace(
        get_running_loop=lambda: SimpleNamespace(time=lambda: now),
        timeout_at=lambda deadline: asyncio.timeout(deadline - now),
        to_thread=asyncio.to_thread,
    ))
    stub = AsyncMock()
    stub.completion.return_value = response(verdict())
    await contract.request_classification(stub, "route", "request", deadline=now + remaining)
    assert 0 < stub.completion.call_args.kwargs["timeout"] <= remaining


async def test_model_compatibility_failure_keeps_the_actionable_reason(monkeypatch):
    from rune.llm.target import ModelCompatibilityError

    reason = "Vertex model gemini-3.1-pro-preview is not available in us-central1. Select a supported location: global."
    stub = client(monkeypatch, ModelCompatibilityError(reason))
    result = await classify_goal("Read the document")
    assert not result.available and reason in result.reason
    assert stub.completion.await_count == 1
    assert from_wire(to_wire(result)).reason == result.reason


@pytest.mark.parametrize("failure", [TimeoutError, httpx.ReadTimeout])
async def test_ambiguous_timeout_is_not_sent_again(monkeypatch, failure):
    stub = client(monkeypatch, failure("secret response body"))
    result = await classify_goal("Save the file")
    assert not result.available and stub.completion.await_count == 1
    assert "secret" not in result.reason and "timeout" in result.reason


async def test_deadline_cancels_one_request_without_running_another(monkeypatch):
    import asyncio

    monkeypatch.setattr("rune.agent.classification_response.ROUTING_TIMEOUT", 0.02)
    cancelled = []

    async def wait(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    stub = client(monkeypatch)
    stub.completion.side_effect = wait
    result = await classify_goal("Open the app")
    assert not result.available and cancelled == [True]
    assert stub.completion.await_count == 1


async def test_routing_uses_selected_model_without_a_weaker_intermediate(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("rune.llm.model_selection.get_effective_model_selection", lambda: SimpleNamespace(
        provider="anthropic", model="claude-opus-5"))
    stub = client(monkeypatch, response(verdict(intent_categories=["desktop"], requires_desktop_input=True)))
    result = await classify_goal("Use Calculator to compute the answer")
    assert result.available and result.requires_desktop_input
    assert stub.completion.await_count == 1
    assert stub.completion.call_args.kwargs["model"] == "claude-opus-5"
    assert stub.completion.call_args.kwargs["provider"] == "anthropic"


@pytest.mark.parametrize("provider,model,effort", [
    ("xai", "grok-4.6", "low"), ("gemini", "gemini-2.5-flash", "none"),
    ("anthropic", "claude-opus-5", "low"), ("openai", "gpt-6-astra", "low"),
    ("xai", "grok-4.3", "none"), ("xai", "unknown-grok", None),
])
async def test_routing_uses_supported_small_effort_without_changing_preferences(monkeypatch, provider, model, effort):
    from types import SimpleNamespace

    from rune.config import get_config

    cfg = get_config()
    original = dict(cfg.llm.reasoning_efforts)
    monkeypatch.setattr("rune.llm.model_selection.get_effective_model_selection", lambda: SimpleNamespace(
        provider=provider, model=model))
    stub = client(monkeypatch, response(verdict()))
    assert (await classify_goal("Read the source file")).available
    assert stub.completion.call_args.kwargs.get("reasoning_effort") == effort
    assert cfg.llm.reasoning_efforts == original


async def test_failed_routing_clears_prior_domain_and_never_selects_tools(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    monkeypatch.setenv('RUNE_HOME', str(tmp_path))
    loop = NativeAgentLoop()
    loop._last_goal_type = 'code_modify'
    select = Mock()
    monkeypatch.setattr(loop, '_select_tools', select)
    bad = ClassificationResult(goal_type='full', confidence=.5, tier=2,
                               available=False, reason='Classification unavailable: read_timeout')
    trace = await loop.run('Open the app', classification=bad)
    assert 'read_timeout' in trace.reason
    assert loop._last_goal_type == ''
    select.assert_not_called()


@pytest.mark.parametrize("intents", [[], ["document"], ["coding"], ["table", "calculation"]])
async def test_local_research_does_not_inherit_web_research(monkeypatch, intents):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.prompts import PROMPT_WEB_BASE, PROMPT_WEB_DEEP

    monkeypatch.setattr("rune.intelligence.repo_map.build_repo_map_sync", lambda *a, **k: "")
    result = ClassificationResult("research", .99, 2, intent_categories=frozenset(intents))
    prompt = await NativeAgentLoop()._build_system_prompt("Inspect supplied files", result)
    assert PROMPT_WEB_DEEP not in prompt and PROMPT_WEB_BASE not in prompt
    contract = resolve_intent_contract(result, .99)
    assert contract.kind == ("code_read" if "coding" in intents else "research")
    assert not contract.requires_code_verification



@pytest.mark.parametrize("complexity,effort", [("simple", "low"), ("moderate", "medium"), ("complex", "high")])
async def test_task_difficulty_reuses_routing_and_survives_handoff(monkeypatch, complexity, effort):
    from rune.agent.goal_classifier import task_reasoning_effort

    stub = client(monkeypatch, response(verdict(goal_type="code_modify", intent_categories=["coding"], complexity=complexity)))
    result = await classify_goal("Continue the change with the original constraints")
    assert result.available and result.complexity == complexity
    assert task_reasoning_effort(result) == effort
    assert task_reasoning_effort(from_wire(to_wire(result))) == effort
    assert stub.completion.await_count == 1


async def test_legacy_routing_does_not_assume_all_work_is_simple(monkeypatch):
    from rune.agent.classification_response import InvalidClassification, validate_decision
    from rune.agent.goal_classifier import ClassificationResult, task_reasoning_effort

    stub = client(monkeypatch, response(verdict()),
                  response(verdict(goal_type="web", lookup_only=True, needs_web_evidence=True)))
    assert (await classify_goal("Continue the work")).complexity == "moderate"
    assert (await classify_goal("Look up the current value")).complexity == "simple"
    assert stub.completion.await_count == 2
    legacy = verdict()
    assert validate_decision(legacy) == legacy
    for invalid in ("unknown", None, [], 3):
        with pytest.raises(InvalidClassification):
            validate_decision(verdict(complexity=invalid))
    task = ClassificationResult("code_modify", .95, 2)
    assert task_reasoning_effort(task) == "medium"
    task.available = False
    assert task_reasoning_effort(task) is None
    task.available, task.confidence = True, .5
    assert task_reasoning_effort(task) is None



@pytest.mark.parametrize("complexity", ["simple", "moderate", "complex"])
@pytest.mark.parametrize("needs_web", [False, True])
def test_task_difficulty_cannot_create_or_remove_a_source_requirement(complexity, needs_web):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.intent_engine import resolve_intent_contract

    task = ClassificationResult("full", .95, 2, complexity=complexity, needs_web_evidence=needs_web)
    contract = resolve_intent_contract(task, task.confidence)
    assert (contract.grounding_requirement == "required") is needs_web
    assert contract.tool_requirement == "read"
