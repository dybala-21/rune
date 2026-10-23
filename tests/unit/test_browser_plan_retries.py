from copy import deepcopy

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.task_evidence import TaskEvidence
from rune.agent.tool_errors import ToolPreparationError
from rune.capabilities.browser.extended import BrowserBatchParams


async def bound_plan(*, explicit=False):
    goal = "Set people to 3 and save" + (". Show Saved" if explicit else "")
    plan = {"items": [
        {"requirement": "people to 3", "kind": "value", "expected": "3"},
        {"requirement": "save. Show Saved" if explicit else "save", "kind": "result",
         "expected": "Saved" if explicit else None},
    ]}
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    await evidence.prepare_requirements()
    evidence.prepare_tool("browser_observe", {"taskRequirements": deepcopy(plan)})
    return evidence, plan


@pytest.mark.parametrize("input_sent", [False, True])
@pytest.mark.parametrize("metadata", [None, "id", "requirement_ids"])
async def test_repeated_plan_keeps_bound_outcomes_and_accepts_matching_references(input_sent, metadata):
    evidence, plan = await bound_plan()
    fixed = evidence.requirements
    evidence.browser_actions = int(input_sent)
    plan["items"][-1]["expected"] = "Saved successfully"
    if metadata:
        for item, bound in zip(plan["items"], fixed.items, strict=True):
            item[metadata] = bound.id if metadata == "id" else [bound.id]
    original = deepcopy(plan)
    evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    assert plan == original and evidence.requirements is fixed
    assert [item.expected for item in fixed.items] == ["3", None]
    assert evidence.acceptance()["status"] != "pass"


@pytest.mark.parametrize("change", ["value", "kind", "missing", "incomplete", "reference", "extra", "explicit"])
async def test_repeated_plan_cannot_replace_or_weaken_the_bound_outcomes(change):
    evidence, plan = await bound_plan(explicit=change == "explicit")
    fixed = evidence.requirements.model_copy(deep=True)
    if change == "value":
        plan["items"][0]["expected"] = "5"
    elif change == "kind":
        plan["items"][0]["kind"] = "preserve"
    elif change == "missing":
        plan["items"].pop(0)
    elif change == "incomplete":
        plan["complete"] = False
    elif change == "reference":
        plan["items"][0]["requirement_ids"] = [fixed.items[1].id]
    elif change == "extra":
        plan["items"][0]["ignore"] = True
    else:
        plan["items"][-1]["expected"] = None
    with pytest.raises(ToolPreparationError):
        evidence.prepare_tool("browser_observe", {"taskRequirements": plan})
    assert evidence.requirements == fixed and evidence.browser_actions == 0


@pytest.mark.parametrize("record_fields", [False, True])
async def test_message_only_inline_readback_becomes_page_observation(record_fields):
    evidence, _ = await bound_plan()
    message = {"selector": "#status", "property": "text", "expected": "Saved successfully", "requirement": "save"}
    fields = [{"selector": "#people", "property": "value", "expected": "3", "requirement": "people to 3"}]
    params = {"actions": [{"type": "act", "params": {
        "action": "click", "selector": "#save",
        "readback": {"url": "https://example.test/form", "checks": [*fields, message] if record_fields else [message]},
    }}]}
    evidence.prepare_tool("browser_batch", params)
    validated = BrowserBatchParams.model_validate(params)
    action = validated.actions[0].params
    if record_fields:
        assert len(action.readback.checks) == 1 and action.readback.checks[0].expected == "3"
    else:
        assert action.readback is None
        assert len(action.checks) == 1 and action.checks[0].expected is None
    assert evidence.acceptance()["status"] != "pass" and evidence.browser_actions == 0
