from unittest.mock import AsyncMock

import pytest

from rune.agent.browser_requirements import request_key
from rune.agent.browser_result import recovery_tools
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.task_evidence import TaskEvidence
from rune.types import CapabilityResult


async def rejected_plan(code="invalid_requirements", **metadata):
    evidence = TaskEvidence.for_request("Set quantity to 3 and save", ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    evidence.observe("browser_navigate", CapabilityResult(success=True, metadata={"document_id": "form"}))
    evidence.observe("browser_batch", CapabilityResult(success=False, metadata={
        "action_status": "not_executed", "validation_error": {"code": code}, **metadata,
    }))
    return evidence


@pytest.mark.parametrize("model", [
    "openai/gpt-5.4-mini", "vertex_ai/gemini-2.5-flash", "anthropic/claude-haiku-4-5", "xai/grok-4",
])
async def test_missing_plan_starts_with_inspection_before_exposing_input(monkeypatch, model):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    evidence = TaskEvidence.for_request("Set quantity to 3 and save", ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    seen = []
    writes = AsyncMock(side_effect=AssertionError("Inspect before input"))

    async def navigate(**kwargs):
        evidence.observe("browser_navigate", CapabilityResult(success=True, metadata={"document_id": "form"}))
        return "Form controls"

    async def complete(**params):
        seen.append(params)
        async def chunks():
            if len(seen) == 1:
                yield _chunk(tool_calls=_tc("browser_navigate"), finish="tool_calls")
            else:
                yield _chunk(content="Page inspected", finish="stop")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    tools = [ToolWrapper(name=name, description=name, function=fn) for name, fn in [
        ("browser_navigate", navigate), ("browser_observe", writes), ("browser_open", writes),
        ("browser_batch", writes), ("browser_act", writes), ("ask_user", writes), ("task_blocked", writes),
    ]]
    agent = LiteLLMAgent(model, tools=tools)
    async with agent.run_stream("Continue", tool_recovery=lambda: recovery_tools(evidence)) as stream:
        assert "".join([part async for part in stream.stream_text()]) == "Page inspected"
    assert len(seen) == 2
    assert seen[0]["tool_choice"] == "required"
    assert {tool["function"]["name"] for tool in seen[0]["tools"]} == {
        "browser_navigate", "browser_open", "browser_observe", "ask_user", "task_blocked",
    }
    assert {"browser_act", "browser_batch"} <= {tool["function"]["name"] for tool in seen[1]["tools"]}
    assert seen[1].get("tool_choice") != "required"
    writes.assert_not_awaited()


async def test_reading_a_page_does_not_force_the_write_recovery_path():
    evidence = TaskEvidence.for_request("Read the page", ClassificationResult("browser", .99, 2, browser_effect="view"))
    await evidence.prepare_requirements()
    assert recovery_tools(evidence) is None


@pytest.mark.parametrize("code", ["requirements_missing", "invalid_requirements", "empty_requirements"])
async def test_rejected_plan_can_be_corrected_without_counting_as_failed_input(code):
    evidence = await rejected_plan(code)
    assert recovery_tools(evidence) == {"browser_observe", "ask_user", "task_blocked"}
    assert evidence.requirements is None
    assert not evidence.browser_failure and evidence.browser_actions == 0


@pytest.mark.parametrize("state", ["denied", "failed_action", "input_sent", "plan_bound", "other_error"])
async def test_plan_recovery_does_not_override_existing_execution(state):
    evidence = await rejected_plan(
        "invalid_arguments" if state == "other_error" else "invalid_requirements",
        **({"approval_decision": "denied"} if state == "denied" else {}),
    )
    if state == "failed_action":
        evidence.browser_failure = "Submission failed"
    elif state == "input_sent":
        evidence.browser_actions = 1
    elif state == "plan_bound":
        evidence.requirements = (await saved_page()).requirements
    assert recovery_tools(evidence) is None
    if state == "denied":
        assert evidence.approval_denied


@pytest.mark.parametrize("model", [
    "openai/gpt-5.4-mini", "vertex_ai/gemini-2.5-flash", "anthropic/claude-haiku-4-5", "xai/grok-4",
])
async def test_plan_recovery_blocks_input_until_corrected_on_each_provider(monkeypatch, model):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    evidence, seen = await rejected_plan(), []
    writes = AsyncMock(side_effect=AssertionError("Input must wait for a valid plan"))

    async def read(**kwargs):
        evidence.prepare_tool("browser_observe", {"taskRequirements": {"items": [
            {"requirement": "quantity to 3", "kind": "value", "expected": "3"},
            {"requirement": "save", "kind": "result", "expected": None},
        ]}})
        evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"document_id": "form"}))
        return "Plan accepted"

    async def complete(**params):
        seen.append(params)
        async def chunks():
            if len(seen) < 3:
                yield _chunk(tool_calls=_tc("browser_batch" if len(seen) == 1 else "browser_observe"), finish="tool_calls")
            else:
                yield _chunk(content="Ready for input", finish="stop")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    tools = [ToolWrapper(name=name, description=name, function=fn) for name, fn in [
        ("browser_observe", read), ("browser_batch", writes), ("browser_act", writes),
        ("ask_user", writes), ("task_blocked", writes),
    ]]
    tools.append(ToolWrapper(name="file_read", description="Read a file", function=writes))
    agent = LiteLLMAgent(model, tools=tools, compact_tools=True, deferred_tools={"file_read"})
    async with agent.run_stream("Continue", tool_recovery=lambda: recovery_tools(evidence)) as stream:
        rejected = await stream._execute_tool("BrowserActChecks", {})
        assert rejected.code == "unknown_tool" and agent._tool_catalog.lookup_required
        assert "".join([part async for part in stream.stream_text()]) == "Ready for input"
    assert len(seen) == 3
    for request in seen[:2]:
        assert request["tool_choice"] == "required"
        assert {tool["function"]["name"] for tool in request["tools"]} == {
            "browser_observe", "ask_user", "task_blocked",
        }
    assert seen[2].get("tool_choice") != "required"
    assert recovery_tools(evidence) is None
    assert evidence.requirements and not evidence.plan_correction_required
    assert not agent._tool_catalog.lookup_required
    writes.assert_not_awaited()


async def saved_page(*, fields_confirmed=True):
    goal = "Set quantity to 3 and save"
    plan = {"complete": True, "items": [
        {"requirement": "quantity to 3", "expected": "3"},
        {"requirement": "save", "kind": "result"},
    ]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult(
        "browser", .99, 2, browser_requirements=plan, browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    field, saved = evidence.requirements.outcomes
    checks = [
        dict(selector="#quantity", property="value", actual="3", expected="3", kind="control",
             passed=True, requirement=field.requirement, requirement_ids=[field.id]),
        dict(selector="#status", property="text", actual="Saved", expected="Saved", kind="result",
             passed=True, requirement=saved.requirement, requirement_ids=[saved.id]),
    ]
    if not fields_confirmed:
        checks = checks[1:]
    evidence.observe("browser_act", CapabilityResult(success=True, metadata={
        "action_status": "dispatched",
        "result_checks": {"passed": True, "effect_observed": True, "checks": checks},
    }))
    return evidence


@pytest.mark.parametrize("state", ["reading", "denied", "failed_action", "no_plan", "partial_page"])
async def test_browser_recovery_does_not_override_other_task_states(state):
    evidence = await saved_page()
    evidence.recovery_requested = True
    if state == "reading":
        evidence.browser_actions = 0
    elif state == "denied":
        evidence.approval_denied = True
    elif state == "failed_action":
        evidence.browser_failure = "Submission rejected"
    elif state == "no_plan":
        evidence.requirements = None
    elif state == "partial_page":
        evidence.covered_requirements.clear()
        evidence.browser_confirmed = False
    assert recovery_tools(evidence) is None


@pytest.mark.parametrize("status,snapshot,expected", [
    ("unavailable", "snapshot", True), ("unavailable", "", False), ("fail", "snapshot", False),
])
async def test_only_correctable_record_checks_keep_recovery_active(status, snapshot, expected):
    evidence = await saved_page()
    evidence.recovery_requested = True
    evidence.independent_readback = {
        "status": status, "snapshot": snapshot, "fields": [{"selector": "#quantity", "property": "value"}],
    }
    assert bool(recovery_tools(evidence)) is expected


async def test_first_saved_object_does_not_end_a_multiple_save_task():
    evidence = await saved_page()
    from rune.agent.browser_requirements import validated

    evidence.request_texts = ("Set quantity to 3 and save the order, then save the note",)
    evidence.requirements = validated({"complete": True, "items": [
        {"requirement": "quantity to 3", "expected": "3"},
        {"requirement": "save the order", "kind": "result"},
        {"requirement": "save the note", "kind": "result"},
    ]}, evidence.request_texts)
    evidence.covered_requirements = {item.id for item in evidence.requirements.outcomes[:2]}
    assert recovery_tools(evidence) is None


@pytest.mark.parametrize("status,snapshot,repair", [
    ("pass", "token", False), ("fail", "", False), ("unavailable", "token", True),
    ("unavailable", "", False),
])
def test_record_guidance_only_requests_a_repair_when_a_snapshot_is_available(status, snapshot, repair):
    from rune.capabilities.browser.readback import readback_result

    report = {"status": status, "checks": [], "fields": [{"selector": "#quantity", "property": "value"}]}
    if snapshot:
        report["snapshot"] = snapshot
    result = readback_result(report)
    assert ("Use browser_observe(readbackSnapshot=" in result.output) is repair


@pytest.mark.parametrize("model", [
    "openai/gpt-5.4-mini", "vertex_ai/gemini-2.5-flash", "anthropic/claude-haiku-4-5", "xai/grok-4",
])
@pytest.mark.parametrize("stored", ["3", "2"])
@pytest.mark.parametrize("fields_confirmed", [True, False])
async def test_recovery_requests_a_read_then_allows_the_measured_result(monkeypatch, model, stored, fields_confirmed):
    from rune.agent.litellm_adapter import LiteLLMAgent
    from rune.agent.tool_adapter import ToolWrapper
    from tests.unit.test_live_streaming import _chunk, _tc, _usage

    evidence = await saved_page(fields_confirmed=fields_confirmed)
    assert (evidence.acceptance()["pageStatus"] == "pass") is fields_confirmed
    assert recovery_tools(evidence) == {"browser_observe", "browser_navigate", "ask_user", "task_blocked"}
    assert not evidence.recovery_requested
    seen, field = [], evidence.requirements.outcomes[0]
    writes = AsyncMock(side_effect=AssertionError("Recovery must not repeat a save"))

    async def read(**kwargs):
        evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"independent_readback": {
            "status": "pass" if stored == "3" else "fail", "scope": "http_document",
            "url": "https://example.test/stored", "result": evidence.requirements.outcomes[1].id,
            "checks": [dict(selector="#quantity", property="value", expected="3", actual=stored,
                            passed=stored == "3", requirement=field.requirement, requirement_ids=[field.id])],
        }}))
        return f"Stored quantity: {stored}"

    async def complete(**params):
        seen.append(params)
        async def chunks():
            if len(seen) == 1:
                yield _chunk(tool_calls=_tc("browser_observe"), finish="tool_calls")
            else:
                yield _chunk(content=f"Stored quantity: {stored}", finish="stop")
            yield _usage()
        return chunks()

    monkeypatch.setattr("litellm.acompletion", complete)
    tools = [ToolWrapper(name=name, description=name, function=fn) for name, fn in [
        ("browser_observe", read), ("browser_navigate", writes), ("browser_act", writes),
        ("ask_user", writes), ("task_blocked", writes),
    ]]
    agent = LiteLLMAgent(model, tools=tools)
    async with agent.run_stream("Confirm the saved quantity", tool_recovery=lambda: recovery_tools(evidence)) as stream:
        assert "".join([part async for part in stream.stream_text()]) == f"Stored quantity: {stored}"
    assert len(seen) == 2
    assert seen[0]["tool_choice"] == "required"
    assert {tool["function"]["name"] for tool in seen[0]["tools"]} == {
        "browser_observe", "browser_navigate", "ask_user", "task_blocked",
    }
    assert seen[1].get("tool_choice") != "required"
    assert recovery_tools(evidence) is None
    assert evidence.acceptance()["status"] == ("pass" if stored == "3" else "fail")
    writes.assert_not_awaited()


@pytest.mark.parametrize("step", [
    {"action": "type", "selector": "#city", "value": "Oslo"},
    {"type": "type", "selector": "#city", "value": "Oslo"},
    {"type": "act", "action": "type", "selector": "#city", "value": "Oslo"},
])
def test_unambiguous_batch_layouts_keep_the_exact_action(step):
    from rune.capabilities.browser.extended import BrowserBatchParams

    plan = BrowserBatchParams(actions=[step])
    assert plan.actions[0].type == "act"
    assert plan.actions[0].params.model_dump(exclude_unset=True) == {
        "action": "type", "selector": "#city", "value": "Oslo",
    }


@pytest.mark.parametrize("step", [
    {"type": "act", "selector": "#city", "value": "Oslo"},
    {"type": "type", "action": "click", "selector": "#city", "value": "Oslo"},
    {"type": "type", "selector": "#city"},
    {"type": "click"},
    {"type": "click", "selector": "#city", "other": "ignored"},
    {"type": "click", "params": {"selector": "#city"}},
    {"type": "act", "params": {"action": "click", "selector": "#city"}, "selector": "#other"},
])
def test_batch_normalization_does_not_guess_or_discard_conflicting_arguments(step):
    from rune.capabilities.browser.extended import BrowserBatchParams

    with pytest.raises(ValueError):
        BrowserBatchParams(actions=[{"action": "click", "selector": "#save"}, step])
