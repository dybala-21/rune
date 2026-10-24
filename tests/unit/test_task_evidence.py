from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.task_evidence import TaskEvidence
from rune.types import CapabilityResult


def contract(goal="browser", **kwargs):
    return resolve_intent_contract(ClassificationResult(goal, .99, 2, **kwargs), .99)


def dispatched(**kwargs):
    return CapabilityResult(success=True, metadata={"action_status": "dispatched", "action": "click", **kwargs})


def checked(passed=True, requirement="Complete the requested task"):
    return {"passed": passed, "effect_observed": True, "checks": [{"property": "text", "expected": "3", "actual": "3" if passed else "1",
                                         "passed": passed, "kind": "result", "requirement": requirement}]}


def test_browser_completion_requires_input_and_readback_without_code_checks():
    c = contract()
    assert c.kind == "browser_write" and not c.requires_code_verification
    evidence = TaskEvidence()
    evidence.observe("browser_observe", CapabilityResult(success=True))
    assert evidence.blocker(c, executions=1)
    evidence.observe("browser_act", dispatched(observation_failed=True))
    assert evidence.blocker(c, executions=0)
    evidence.observe("browser_observe", CapabilityResult(success=True))
    assert evidence.blocker(c, executions=0)
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": checked()}))
    assert evidence.blocker(c, executions=0) is None


def test_failed_batch_is_not_repaired_by_observation_or_scroll():
    e = TaskEvidence()
    e.observe("browser_batch", CapabilityResult(success=False, error="stale target", metadata={"browser_steps": [
        {"tool": "browser_act", "success": True, "metadata": dispatched().metadata},
        {"tool": "browser_act", "success": False, "error": "stale target"},
    ]}))
    e.observe("browser_observe", CapabilityResult(success=True))
    e.observe("browser_act", dispatched(action="scroll"))
    assert e.blocker(contract(), 0)
    e.observe("browser_act", dispatched(result_checks=checked()))
    assert e.blocker(contract(), 0) is None


def test_requested_scroll_can_complete_after_observation():
    e = TaskEvidence(browser_effect="view")
    e.observe("browser_act", dispatched(action="scroll"))
    assert e.blocker(contract(), 0) is None


@pytest.mark.parametrize("tool", ["browser_act", "browser_navigate", "browser_open", "browser_observe", "browser_extract"])
def test_old_page_checks_do_not_verify_later_state(tool):
    evidence = TaskEvidence()
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": checked()}))
    assert evidence.blocker(contract(), 0) is None
    evidence.observe(tool, dispatched())
    assert evidence.blocker(contract(), 0)


def test_calculation_uses_structured_fresh_evidence_and_does_not_satisfy_a_command():
    e = TaskEvidence()
    c = contract("research", intent_categories=frozenset({"calculation"}))
    assert c.kind == "calculation"
    e.observe("file_read", CapabilityResult(success=True, output="sum=37500"))
    assert e.blocker(c, 0)
    e.observe("file_read", CapabilityResult(success=True, metadata={
        "cached": True, "table_profile": {"complete": True, "rows": 3},
    }))
    assert e.blocker(c, 0)
    e.observe("file_read", CapabilityResult(success=True, metadata={
        "table_profile": {"complete": True, "rows": 3, "sums": {"amount": "37500"}},
    }))
    assert e.blocker(c, 0) is None
    command = contract("execution", requires_execution=True, intent_categories=frozenset({"calculation"}))
    assert command.kind == "execution" and command.tool_requirement == "execute"
    assert not command.requires_code_write_artifact


def test_web_calculation_requires_a_successful_query_receipt():
    evidence = TaskEvidence()
    target = contract("research", intent_categories=frozenset({"calculation"}))
    metadata = {"record_query": {"scope": "retrieved_response", "matched": 2, "query": {"operation": "sum"}}}
    evidence.observe("web_fetch", CapabilityResult(success=False, metadata=metadata))
    assert evidence.blocker(target, 0)
    evidence.observe("web_fetch", CapabilityResult(success=True, metadata={**metadata, "cached": True}))
    assert evidence.blocker(target, 0)
    evidence.observe("web_fetch", CapabilityResult(success=True, metadata=metadata))
    assert evidence.blocker(target, 0) is None


def test_unrelated_command_does_not_verify_a_calculation():
    evidence = TaskEvidence(calculation_operation="sum")
    target = contract("research", intent_categories=frozenset({"calculation"}))
    evidence.observe("bash_execute", CapabilityResult(success=True, output="/workspace"))
    assert evidence.blocker(target, 1)


@pytest.mark.parametrize("command,computed", [
    ("pwd", False), ("ls -la data", False),
    ("python3 -c \"import openpyxl; print(sum(...))\"", True), ("cd data && uv run python mean.py", True),
])
def test_only_code_that_computes_counts_as_a_calculation(command, computed):
    """An .xlsx average computed with Python is a result; a listing is not."""
    evidence = TaskEvidence(calculation_operation="mean")
    target = contract("research", intent_categories=frozenset({"calculation"}))
    evidence.observe("bash_execute", CapabilityResult(success=True, output="41.5"), {"command": command})
    assert (evidence.blocker(target, 1) is None) is computed
    failed = TaskEvidence(calculation_operation="mean")
    failed.observe("bash_execute", CapabilityResult(success=False, error="Traceback"), {"command": command})
    assert failed.blocker(target, 1)


def test_checks_must_cover_prior_expectations_and_quote_the_request():
    goal = "Set the quantity to 3 and save"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    expected = checked(requirement="quantity to 3")
    evidence.prepare("browser_act", {"checks": expected["checks"]})
    unrelated = checked(requirement="a requirement invented by the model")
    unrelated["checks"][0].update(expected="unrelated", actual="unrelated")
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": unrelated}))
    assert evidence.blocker(contract(), 0)
    weakened = checked(requirement="quantity to 3")
    weakened["checks"][0].update(expected="1", actual="1")
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": weakened}))
    assert evidence.blocker(contract(), 0)


def test_unsaved_control_is_enough_for_edit_but_not_submission():
    goal = "Set the quantity to 3"
    receipt = checked(requirement=goal)
    receipt["effect_observed"] = False
    receipt["checks"][0].update(kind="control", property="value")
    for effect in ("edit", "change"):
        evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2, browser_effect=effect))
        evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": receipt}))
        assert (evidence.blocker(contract(), 0) is None) == (effect == "edit")


def test_confirmation_wording_can_be_corrected_without_weakening_requested_values():
    goal = "도시를 군산, 인원을 3으로 바꾸고 저장해"
    evidence = TaskEvidence.for_request(goal, ClassificationResult("browser", .99, 2))
    checks = [{"property": "value", "expected": "군산", "actual": "군산", "passed": True,
               "kind": "control", "requirement": "도시를 군산으로 변경"},
              {"property": "text", "expected": "Saved", "actual": "저장됨", "passed": False,
               "kind": "result", "requirement": "저장해"}]
    evidence.observe("browser_act", dispatched(result_checks={"passed": False, "checks": checks}))
    assert evidence.blocker(contract(), 0)
    checks[1].update(expected="저장됨", passed=True)
    evidence.observe("browser_observe", CapabilityResult(success=True, metadata={"result_checks": {"passed": True, "effect_observed": True, "checks": checks}}))
    assert evidence.blocker(contract(), 0) is None


def test_batch_readback_carries_requested_values_without_changing_expectations():
    evidence = TaskEvidence.for_request("Set count to 3 and save", ClassificationResult("browser", .99, 2))
    field = {"selector": "#count", "property": "value", "expected": "3", "requirement": "count to 3"}
    confirmation = {"selector": "#saved", "property": "text", "expected": "Saved", "requirement": "save"}
    params = {"actions": [{"type": "act", "params": {"checks": [field]}},
                           {"type": "observe", "params": {"checks": [confirmation]}}]}
    evidence.prepare("browser_batch", params)
    assert params["actions"][1]["params"]["checks"] == [confirmation, field]
    changed = {"checks": [{**field, "expected": "2"}]}
    evidence.prepare("browser_observe", changed)
    assert field in changed["checks"]
    evidence.prepare("browser_batch", {"actions": None})


def test_malformed_selector_cannot_break_evidence_preparation():
    evidence = TaskEvidence(request_texts=("Set count to 3",))
    evidence.prepare("browser_observe", {"checks": [{
        "selector": ["#count"], "property": "value", "expected": "3", "requirement": "Set count to 3",
    }]})
    assert not evidence.expected_checks


async def test_schema_rejection_does_not_register_completion_requirements():
    from unittest.mock import AsyncMock

    from rune.agent.tool_adapter import ToolAdapterOptions, _build_typed_tool
    from rune.capabilities.browser.capabilities import BrowserActParams
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition

    evidence = TaskEvidence(request_texts=("Set quantity to 3 and save",))
    execute = AsyncMock(return_value=CapabilityResult(success=True))
    cap = CapabilityDefinition(name="browser_act", description="act", execute=execute, parameters_model=BrowserActParams)
    registry = CapabilityRegistry()
    registry.register(cap)
    tool = _build_typed_tool(cap_def=cap, reg=registry, cache=None, stall=None,
                            opts=ToolAdapterOptions(prepare_tool=evidence.prepare))
    result = await tool.function(action="fill", selector="#quantity", value="2", checks=[
        {"selector": "#quantity", "property": "value", "expected": "2", "requirement": "quantity to 3"}])
    execute.assert_not_awaited()
    assert "Invalid arguments" in result
    assert not evidence.expected_checks


def test_previously_read_field_cannot_be_replaced_after_disappearing():
    goal = "Set both fields to 3"
    evidence = TaskEvidence(request_texts=(goal,))
    field = {"selector": "#first", "property": "value", "expected": "3", "requirement": goal}
    for error in ("", "Target missing"):
        evidence.observe("browser_observe", CapabilityResult(success=True, metadata={
            "result_checks": {"passed": False, "checks": [{**field, "actual": "9", "passed": False, "error": error}]},
        }))
    replacement = {"checks": [{**field, "selector": "#second"}]}
    evidence.prepare("browser_observe", replacement)
    assert replacement["checks"] == [{**field, "selector": "#second"}, field]


@pytest.mark.parametrize("case,completed", [("observe", False), ("act", True), ("failed", False),
                                           ("incomplete_plan", False), ("unavailable_plan", False),
                                           ("denied", False), ("rejected_calls", False),
                                           ("scroll_instead_of_save", False), ("count_instead_of_sum", False),
                                           ("readback", True), ("failed_readback", False),
                                           ("unbound_result", False), ("missing_baseline", False), ("missing_second_action", False),
                                           ("linked_confirmation", False), ("fetched_pass", True),
                                           ("fetched_fail", False), ("fetched_unavailable", False),
                                           ("fetched_budget_pass", True), ("fetched_budget_fail", False),
                                           ("fetched_budget_unavailable", False), ("fetched_budget_missing", False),
                                           ("fetched_budget_document", False), ("fetched_budget_cancelled", False),
                                           ("profile", True), ("document_profile", True), ("raw_csv", False), ("command", False),
                                           ("local_calculation", True), ("invalid_calculation", False),
                                           ("unrequested_calculation", False), ("unrelated_command", False)])
async def test_real_loop_does_not_finalize_without_requested_evidence(monkeypatch, tmp_path, case, completed):
    from rune.agent.loop import NativeAgentLoop

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    options = []
    rows = [("browser_observe", CapabilityResult(success=True, output="Previous quote: 51000"))]
    classification = ClassificationResult("browser", .99, 2)
    goal = "Complete the requested task"
    if case == "denied":
        classification = ClassificationResult("execution", .99, 2, requires_execution=True)
        rows = [("web_fetch", CapabilityResult(success=False, error="User declined the network operation.", metadata={
            "action_status": "not_executed", "approval_decision": "denied",
        }))]
    elif case == "act":
        rows.append(("browser_act", dispatched(result_checks=checked())))
    elif case == "scroll_instead_of_save":
        goal = "Set quantity to 3 and save."
        rows.append(("browser_act", dispatched(action="scroll")))
    elif case in {"readback", "failed_readback"}:
        goal = "Finish the quantity change from the earlier task."
        rows = [("browser_observe", CapabilityResult(success=True, metadata={"result_checks": checked(case == "readback", goal)}))]
    elif case == "count_instead_of_sum":
        goal = "Sum the amounts 10 and 20."
        classification = ClassificationResult("research", .99, 2, intent_categories=frozenset({"calculation"}), calculation_operation="sum")
        rows = [("web_fetch", CapabilityResult(success=True, output='{"count":2}', metadata={
            "record_query": {"scope": "retrieved_response", "matched": 2, "query": {"operation": "count"}},
        }))]
    elif case == "unrelated_command":
        goal = "Calculate the sum of amounts in orders.csv"
        classification = ClassificationResult("research", .99, 2, intent_categories=frozenset({"calculation"}), calculation_operation="sum")
        rows = [("bash_execute", CapabilityResult(success=True, output="/workspace"))]
    elif case == "failed":
        rows += [("browser_act", dispatched()), ("browser_act", CapabilityResult(success=False, error="not loaded"))]
    elif case in {"profile", "document_profile", "raw_csv", "command"}:
        classification = ClassificationResult("research", .99, 2, intent_categories=frozenset({"calculation"}))
        metadata = {"table_profile": {"complete": True, "rows": 3, "sums": {"amount": "37500"}}} if case != "raw_csv" else {}
        reader = "document_read" if case == "document_profile" else "file_read"
        rows = [(reader, CapabilityResult(success=True, output="3 rows; amount sum=37500", metadata=metadata))]
        if case == "command":
            classification = ClassificationResult("execution", .99, 2, requires_execution=True)
    elif case in {"local_calculation", "invalid_calculation", "unrequested_calculation"}:
        goal = "173 × 29 − 417의 값을 숫자만으로 답해줘."
        expression = "173 × 29 − 417" if case != "invalid_calculation" else "1 / 0"
        if case == "unrequested_calculation":
            goal = "Read the supplied data and calculate the total."
        elif case == "invalid_calculation":
            goal = "Calculate 1 / 0."
        classification = ClassificationResult("chat", .99, 2, intent_categories=frozenset({"calculation"}),
                                                calculation_expression=expression)
        rows = []

    class Stream:
        tool_budget_exhausted = False
        _task_blocked = "Stopped after 3 rejected calls; no input was dispatched." if case == "rejected_calls" else ""
        async def stream_text(self, **kwargs):
            for name, result in rows:
                await options[-1].on_tool_start(name, {})
                await options[-1].on_tool_end(name, result)
            if case.startswith("fetched_budget_"):
                if case == "fetched_budget_cancelled":
                    loop._cancel_event.set()
                from rune.agent.request_budget import BudgetExceeded
                raise BudgetExceeded("The model request limit was reached.")
            yield "The available result is shown above."
        async def get_output(self): return "The available result is shown above."
        def usage(self): return SimpleNamespace(input_tokens=10, output_tokens=10)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return []

    turns = []

    class Agent:
        def __init__(self, *args, **kwargs): pass
        @asynccontextmanager
        async def run_stream(self, *args, **kwargs):
            turns.append(kwargs)
            yield Stream()

    def build(opts):
        options.append(opts)
        return {}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    from rune.agent.browser_requirements import request_key
    classification.browser_requirements = {"complete": True, "items": [{"requirement": goal, "expected": None}]}
    classification.browser_requirements_key = request_key((goal,))
    if case == "incomplete_plan":
        classification.browser_requirements["complete"] = False
    elif case == "unavailable_plan":
        from unittest.mock import AsyncMock
        classification.browser_requirements = None
        monkeypatch.setattr("rune.llm.client.get_llm_client", AsyncMock(return_value=None))
    elif case in {"unbound_result", "missing_baseline", "missing_second_action"}:
        goal = "Save once and confirm the save" if case == "unbound_result" else "Save once and send to Alex"
        unresolved = "confirm the save" if case == "unbound_result" else "send to Alex"
        classification.browser_requirements = {"complete": True, "items": [
            {"requirement": "Save once", "kind": "result"}, {"requirement": unresolved, "kind": "result"},
        ]}
        if case == "missing_baseline":
            goal = "Save once and preserve the reference"
            classification.browser_requirements["items"][1] = {"requirement": "preserve the reference", "kind": "preserve"}
        classification.browser_requirements_key = request_key((goal,))
        rows = [("browser_act", dispatched(result_checks=checked(requirement="Save once")))]
    elif case == "linked_confirmation" or case.startswith("fetched_"):
        goal = "Set count to 3. Save once and confirm the save"
        classification.browser_requirements = {"complete": True, "items": [
            {"requirement": "Save once", "kind": "result"},
            {"requirement": "confirm the save", "kind": "result", "confirms": "Save once"},
        ]}
        classification.browser_requirements_key = request_key((goal,))
        rows = [("browser_act", dispatched(result_checks=checked(requirement="Save once")))]
        if case.startswith("fetched_"):
            classification.browser_requirements["items"].append({"requirement": "count to 3", "expected": "3"})
            from rune.agent.browser_requirements import BrowserRequirements
            plan = BrowserRequirements.model_validate(classification.browser_requirements)
            rows.append(("browser_observe", CapabilityResult(success=True, metadata={
                "independent_readback": {"status": case.removeprefix("fetched_").removeprefix("budget_")
                    if case.endswith(("pass", "fail", "unavailable")) else "pass", "scope": "http_document",
                    "url": "https://example.test/record", "result": plan.items[0].id,
                    "checks": [{"requirement_ids": [plan.items[-1].id], "selector": "#count", "property": "value",
                    "expected": "3", "actual": "3", "passed": True}]},
            })))
            if case == "fetched_budget_missing":
                goal += " and notify Alex"
                classification.browser_requirements["items"].append({"requirement": "notify Alex", "kind": "result"})
                classification.browser_requirements_key = request_key((goal,))
            if case == "fetched_budget_document":
                classification.intent_categories = frozenset({"document"})
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    trace = await loop._execute_loop(goal, "", [], 2, classification)
    assert (trace.reason == "completed") is completed
    if case == "linked_confirmation" or case.startswith("fetched_"):
        from rune.agent.run_outcome import run_outcome
        assert len(turns) == (2 if case == "linked_confirmation" else 1)
        assert run_outcome(trace).success is completed
        if case == "fetched_budget_pass":
            assert "3" in loop._last_answer_text
            assert "budget" in loop._last_answer_text.lower() or "한도" in loop._last_answer_text
    if case == "denied":
        assert trace.reason == "task_blocked" and len(turns) == 1
    if case == "rejected_calls":
        from rune.agent.run_outcome import run_outcome

        assert trace.reason == "task_incomplete" and len(turns) == 1
        assert trace.completion_check["name"] == "Tool execution"
        assert not run_outcome(trace).success
    if case in {"incomplete_plan", "unavailable_plan"}:
        assert len(turns) == 1 and trace.reason == "verification_unavailable"
        assert trace.requirement_acceptance["status"] == "unavailable"
    if case in {"unbound_result", "missing_baseline", "missing_second_action"}:
        from rune.agent.run_outcome import run_outcome

        assert len(turns) == 1 and trace.reason == "verification_inconclusive"
        assert trace.requirement_acceptance["readback"]["status"] == "pass"
        assert run_outcome(trace).completion == "unknown"
        assert not run_outcome(trace).success


def test_tool_selection_excludes_unrelated_work():
    from rune.agent.loop import NativeAgentLoop
    loop = NativeAgentLoop()
    for goal, intents in [("chat", frozenset()), ("browser", frozenset()), ("research", frozenset({"calculation"}))]:
        tools = loop._select_tools(ClassificationResult(goal, .99, 2, intent_categories=intents))
        assert not {"document_create", "table_requirements", "table_verify", "file_write"} & set(tools)
        if goal in {"chat", "browser"}:
            assert "file_read" not in tools


@pytest.mark.parametrize("goal", ["code_modify", "full"])
def test_calculation_label_does_not_remove_deliverable_tools(goal):
    from rune.agent.loop import NativeAgentLoop

    classification = ClassificationResult(goal, .9, 2, requires_execution=False,
                                          intent_categories=frozenset({"calculation", "table"}))
    tools = NativeAgentLoop()._select_tools(classification)
    assert {"file_write", "table_requirements", "table_verify"} <= set(tools)
