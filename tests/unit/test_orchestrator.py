"""Tests for the multi-agent orchestrator module."""

from __future__ import annotations

import pytest

from rune.agent.orchestrator import (
    OrchestrationPlan,
    Orchestrator,
    OrchestratorConfig,
    SubTask,
    SubTaskResult,
)


class TestFailureClassification:
    def test_classify_critical_permission(self):
        assert Orchestrator._classify_failure("Permission denied") == "critical"

    def test_classify_critical_oom(self):
        assert Orchestrator._classify_failure("Out of memory error") == "critical"

    def test_classify_critical_segfault(self):
        assert Orchestrator._classify_failure("Segmentation fault") == "critical"

    def test_classify_systemic_module_not_found(self):
        assert Orchestrator._classify_failure("Module not found: foo") == "systemic"

    def test_classify_systemic_command_not_found(self):
        assert Orchestrator._classify_failure("command not found: npm") == "systemic"

    def test_classify_transient_timeout(self):
        assert Orchestrator._classify_failure("Request timeout after 30s") == "transient"

    def test_classify_transient_rate_limit(self):
        assert Orchestrator._classify_failure("429 rate limit exceeded") == "transient"

    def test_classify_transient_network(self):
        assert Orchestrator._classify_failure("Network connection reset") == "transient"

    def test_classify_correctable_default(self):
        assert Orchestrator._classify_failure("Something unexpected happened") == "correctable"


class TestMergeResults:
    def test_merge_empty(self):
        assert Orchestrator._merge_results([]) == ""

    def test_merge_success(self):
        results = [
            SubTaskResult(task_id="t1", success=True, output="output A"),
            SubTaskResult(task_id="t2", success=True, output="output B"),
        ]
        merged = Orchestrator._merge_results(results)
        assert "output A" in merged
        assert "output B" in merged

    def test_merge_with_failure(self):
        results = [
            SubTaskResult(task_id="t1", success=True, output="ok"),
            SubTaskResult(task_id="t2", success=False, error="crashed"),
        ]
        merged = Orchestrator._merge_results(results)
        assert "ok" in merged
        assert "FAILED" in merged
        assert "crashed" in merged


class TestParsePlanJson:
    def test_valid_plan_json(self):
        raw = '{"tasks": [{"id": "a", "description": "do A"}, {"id": "b", "description": "do B", "dependencies": ["a"]}]}'
        plan = Orchestrator._try_parse_plan_json(raw)
        assert plan is not None
        assert len(plan.tasks) == 2
        assert plan.tasks[1].dependencies == ["a"]

    def test_invalid_json(self):
        plan = Orchestrator._try_parse_plan_json("not json at all")
        assert plan is None

    def test_json_without_tasks(self):
        plan = Orchestrator._try_parse_plan_json('{"description": "no tasks"}')
        assert plan is None

    def test_json_embedded_in_text(self):
        raw = 'Here is the plan: {"tasks": [{"id": "x", "description": "task x"}]} -- end'
        plan = Orchestrator._try_parse_plan_json(raw)
        assert plan is not None
        assert len(plan.tasks) == 1


class TestOrchestratorExecution:
    @pytest.mark.asyncio
    async def test_execute_single_task_stub(self):
        """Without an agent loop factory, the orchestrator uses stub execution."""
        orch = Orchestrator(config=OrchestratorConfig(max_workers=1, risk_gate_enabled=False))
        plan = OrchestrationPlan(
            tasks=[SubTask(id="t1", description="Do something")],
        )
        result = await orch.execute("test goal", plan=plan)
        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].success is True
        assert "[stub]" in result.results[0].output

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self):
        """Tasks with dependencies should still complete in order."""
        orch = Orchestrator(config=OrchestratorConfig(max_workers=2, risk_gate_enabled=False))
        plan = OrchestrationPlan(
            tasks=[
                SubTask(id="t1", description="First task"),
                SubTask(id="t2", description="Second task", dependencies=["t1"]),
            ],
        )
        result = await orch.execute("test", plan=plan)
        assert result.success is True
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_execute_risk_gate_blocks_critical(self):
        """Critical risk plans should be blocked."""
        orch = Orchestrator(config=OrchestratorConfig(risk_gate_enabled=True))
        plan = OrchestrationPlan(
            tasks=[SubTask(id="t1", description="Dangerous")],
            risk_level="critical",
        )
        result = await orch.execute("dangerous goal", plan=plan)
        assert result.success is False
        assert "rejected" in result.merged_output.lower()

    @pytest.mark.asyncio
    async def test_execute_auto_generates_plan(self):
        """Without an explicit plan, a single-task fallback plan is generated."""
        orch = Orchestrator(config=OrchestratorConfig(risk_gate_enabled=False))
        result = await orch.execute("do something useful")
        assert result.success is True
        assert len(result.results) == 1
