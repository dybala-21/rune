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


def test_browser_completion_requires_input_and_readback_without_code_checks():
    c = contract()
    assert c.kind == "browser_write" and not c.requires_code_verification
    evidence = TaskEvidence()
    evidence.observe("browser_observe", CapabilityResult(success=True))
    assert evidence.blocker(c, executions=1)
    evidence.observe("browser_act", dispatched(observation_failed=True))
    assert evidence.blocker(c, executions=0)
    evidence.observe("browser_observe", CapabilityResult(success=True))
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
    e.observe("browser_act", dispatched())
    assert e.blocker(contract(), 0) is None


def test_requested_scroll_can_complete_after_observation():
    e = TaskEvidence()
    e.observe("browser_act", dispatched(action="scroll"))
    assert e.blocker(contract(), 0) is None


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


@pytest.mark.parametrize("case,completed", [("observe", False), ("act", True), ("failed", False),
                                           ("profile", True), ("document_profile", True), ("raw_csv", False), ("command", False),
                                           ("local_calculation", True), ("invalid_calculation", False),
                                           ("unrequested_calculation", False)])
async def test_real_loop_does_not_finalize_without_requested_evidence(monkeypatch, tmp_path, case, completed):
    from rune.agent.loop import NativeAgentLoop

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    options = []
    rows = [("browser_observe", CapabilityResult(success=True, output="Previous quote: 51000"))]
    classification = ClassificationResult("browser", .99, 2)
    goal = "Complete the requested task"
    if case == "act":
        rows.append(("browser_act", dispatched()))
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
        async def stream_text(self, **kwargs):
            for name, result in rows:
                await options[-1].on_tool_start(name, {})
                await options[-1].on_tool_end(name, result)
            yield "The available result is shown above."
        async def get_output(self): return "The available result is shown above."
        def usage(self): return SimpleNamespace(input_tokens=10, output_tokens=10)
        def get_failure_state(self): return {}, set()
        def all_messages(self): return []

    class Agent:
        def __init__(self, *args, **kwargs): pass
        @asynccontextmanager
        async def run_stream(self, *args, **kwargs): yield Stream()

    def build(opts):
        options.append(opts)
        return {}

    monkeypatch.setattr("rune.agent.loop.build_tool_set", build)
    monkeypatch.setattr("rune.agent.loop.LiteLLMAgent", Agent)
    loop = NativeAgentLoop()
    loop._workspace_root = str(tmp_path)
    trace = await loop._execute_loop(goal, "", [], 2, classification)
    assert (trace.reason == "completed") is completed


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
