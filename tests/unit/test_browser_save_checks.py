import pytest


@pytest.mark.parametrize("goal,quote,wording", [
    ("인원을 3으로 바꾸고 저장됐는지도 확인해.", "저장됐는지도 확인해.", "저장되었습니다."),
    ("Set people to 3 and save.", "save", "Saved successfully"),
    ("Personen auf 3 ändern und speichern.", "speichern", "Gespeichert"),
])
async def test_executor_does_not_turn_generated_success_text_into_a_requirement(goal, quote, wording):
    from copy import deepcopy

    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.task_evidence import TaskEvidence

    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    plan = {"items": [
        {"requirement": "3", "kind": "value", "expected": "3"},
        {"requirement": quote, "kind": "result", "expected": wording},
    ]}
    original = deepcopy(plan)
    evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    assert plan == original
    assert evidence.requirements.complete
    assert [item.expected for item in evidence.requirements.items] == ["3", None]
    assert evidence.acceptance()["status"] != "pass"


@pytest.mark.parametrize("case", ["explicit", "other_quote", "field", "paraphrase", "incomplete", "after_input", "invalid_items"])
async def test_result_wording_cleanup_preserves_requested_conditions_and_execution_guards(case):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.task_evidence import TaskEvidence
    from rune.agent.tool_errors import ToolPreparationError

    goal = "Set people to 3 and save"
    item = {"requirement": "save", "kind": "result", "expected": "Saved"}
    plan = {"items": [{"requirement": "people to 3", "kind": "value", "expected": "3"}, item]}
    if case in {"explicit", "other_quote"}:
        goal += ". Show Saved"
        if case == "explicit":
            item["requirement"] = "save. Show Saved"
    elif case == "field":
        item["kind"] = "value"
    elif case == "paraphrase":
        item["requirement"] = "confirm saving"
    elif case == "incomplete":
        plan["complete"] = False
    elif case == "invalid_items":
        plan["items"] = None
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    if case == "after_input":
        evidence.browser_actions = 1
    if case in {"explicit", "incomplete"}:
        evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
        if case == "explicit":
            assert evidence.requirements.items[-1].expected == "Saved"
        else:
            assert not evidence.requirements.complete
            with pytest.raises(ToolPreparationError, match="requirements_unclear"):
                evidence.prepare_tool("browser_act", {"action": "click", "selector": "#save"})
    else:
        with pytest.raises(ToolPreparationError):
            evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
        assert evidence.requirements is None

from rune.agent.browser_requirements import request_key
from rune.agent.goal_classifier import ClassificationResult
from rune.agent.task_evidence import TaskEvidence


@pytest.mark.parametrize("expected", [None, "Saved"])
async def test_save_action_does_not_require_invented_message(expected):
    outcome = "save" if expected is None else "save. Show Saved"
    goal = "Set count to 3 and " + outcome
    plan = {"complete": True, "items": [
        {"requirement": "count to 3", "expected": "3"},
        {"requirement": outcome, "kind": "result", "expected": expected},
    ]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult(
        "browser", .99, 2, browser_requirements=plan, browser_requirements_key=request_key((goal,)),
    ))
    await evidence.prepare_requirements()
    params = {"action": "click", "selector": "#save", "checks": [
        {"selector": "#count", "property": "value", "expected": "3", "requirement": "count to 3"},
        {"selector": "#message", "property": "text", "expected": "Saved", "requirement": outcome},
    ]}
    evidence.prepare_tool("browser_act", params)
    assert len(params["checks"]) == 2
    assert params["checks"][1]["expected"] == expected
    assert evidence.acceptance()["status"] != "pass"
    retry = {"action": "click", "selector": "#save"}
    evidence.prepare_tool("browser_act", retry)
    assert len(retry["checks"]) == (1 if expected is None else 2)


@pytest.mark.parametrize("property", ["value", "checked", "url"])
def test_field_checks_cannot_drop_their_expected_value(property):
    from pydantic import ValidationError

    from rune.capabilities.browser.checks import BrowserCheck

    with pytest.raises(ValidationError):
        BrowserCheck(selector="#field", property=property, expected=None)


async def test_observed_message_remains_available_as_page_evidence():
    goal = "save"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2,
        browser_requirements={"complete": True, "items": [{"requirement": goal, "kind": "result"}]},
        browser_requirements_key=request_key((goal,))))
    await evidence.prepare_requirements()
    params = {"checks": [{"selector": "#message", "property": "text", "expected": "Saved", "requirement": goal}]}
    evidence.prepare_tool("browser_observe", params)
    assert len(params["checks"]) == 1
    assert evidence.acceptance()["status"] != "pass"


@pytest.mark.parametrize("locale", ["en", "ko"])
async def test_observed_message_is_not_described_as_user_requested_text(locale):
    from rune.agent.browser_result import record_facts

    evidence = TaskEvidence.for_request("save", ClassificationResult("browser", .99, 2,
        browser_requirements={"complete": True, "items": [{"requirement": "save", "kind": "result"}]},
        browser_requirements_key=request_key(("save",))))
    await evidence.prepare_requirements()
    evidence.page_checks = [{"requirement": "save", "property": "text", "expected": "Saved", "actual": "Saved"}]
    summary = record_facts(evidence, locale)
    assert "Saved" in summary
    assert "requested:" not in summary and "요청:" not in summary


async def test_first_batch_can_bind_requirements_after_page_inspection():
    from rune.agent.browser_plan import configure_tools
    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import register_browser_capabilities
    from rune.capabilities.registry import CapabilityRegistry

    goal = "Set count to 3 and save"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    registry = CapabilityRegistry()
    register_browser_capabilities(registry)
    tools = {name: _build_typed_tool(cap_def=registry.get(name), reg=registry, cache=None, stall=None,
                                    opts=ToolAdapterOptions(prepare_tool=evidence.prepare_tool))
             for name in ("browser_navigate", "browser_act", "browser_batch")}
    configure_tools(evidence, tools)
    assert "taskRequirements" not in tools["browser_navigate"].json_schema["properties"]
    assert "taskRequirements" in tools["browser_batch"].json_schema["properties"]
    params = {"taskRequirements": {"items": [
        {"requirement": "count to 3", "expected": "3", "kind": "value"},
        {"requirement": "save", "expected": None, "kind": "result"},
    ]}, "actions": [{"type": "act", "params": {"action": "type", "selector": "#count", "value": "3"}}]}
    evidence.prepare_tool("browser_batch", params)
    assert evidence.requirements.complete and evidence.browser_actions == 0
    assert "taskRequirements" not in params
