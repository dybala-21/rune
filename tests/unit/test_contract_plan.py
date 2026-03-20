"""Tests for rune.agent.contract_plan — contract-based planning."""


from rune.agent.contract_plan import (
    BuildContractPlanInput,
    build_contract_plan_trace,
    build_probe_candidates,
)
from rune.agent.intent_engine import IntentContract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_input(**kwargs) -> BuildContractPlanInput:
    defaults = dict(
        goal="implement feature",
        intent=IntentContract(
            kind="code_write",
            tool_requirement="write",
            grounding_requirement="none",
            output_expectation="file",
            requires_code_verification=True,
        ),
        intent_resolved=True,
        workspace_root="/tmp",
        workspace_files=[],
    )
    defaults.update(kwargs)
    return BuildContractPlanInput(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildContractPlanTrace:
    def test_write_plan_with_go_verification(self):
        plan = build_contract_plan_trace(make_input(
            goal="handle gateway timeout and finish tests",
            workspace_files=["go.mod"],
        ))
        assert len(plan.action_plan) >= 3
        assert "go test ./..." in plan.verification_candidates

    def test_grounding_step_when_required(self):
        plan = build_contract_plan_trace(make_input(
            goal="summarize latest regulations",
            intent=IntentContract(
                kind="research",
                tool_requirement="read",
                grounding_requirement="required",
                output_expectation="text",
                requires_code_verification=False,
            ),
        ))
        assert any("source" in step.lower() or "verify" in step.lower() for step in plan.action_plan)
        assert any("source" in c.lower() or "required" in c.lower() for c in plan.completion_criteria)

    def test_unresolved_intent_adds_criterion(self):
        plan = build_contract_plan_trace(make_input(
            goal="ambiguous request",
            intent_resolved=False,
            intent=IntentContract(
                kind="mixed",
                tool_requirement="write",
                grounding_requirement="none",
                output_expectation="either",
                requires_code_verification=False,
            ),
        ))
        assert any("resolved" in c.lower() for c in plan.completion_criteria)

    def test_non_code_write_has_no_verification_candidates(self):
        plan = build_contract_plan_trace(make_input(
            goal="write a research report",
            workspace_files=["package.json"],
            intent=IntentContract(
                kind="mixed",
                tool_requirement="write",
                grounding_requirement="none",
                output_expectation="file",
                requires_code_verification=False,
            ),
        ))
        assert len(plan.verification_candidates) == 0

    def test_python_project_verification(self):
        plan = build_contract_plan_trace(make_input(
            workspace_files=["pyproject.toml"],
        ))
        assert "pytest" in plan.verification_candidates

    def test_node_project_verification(self):
        plan = build_contract_plan_trace(make_input(
            workspace_files=["package.json"],
        ))
        assert "npm run test" in plan.verification_candidates

    def test_cargo_project_verification(self):
        plan = build_contract_plan_trace(make_input(
            workspace_files=["Cargo.toml"],
        ))
        assert "cargo test" in plan.verification_candidates

    def test_objective_is_clipped(self):
        long_goal = "x " * 200
        plan = build_contract_plan_trace(make_input(goal=long_goal))
        assert len(plan.objective) <= 143  # 140 + "..."

    def test_read_only_intent_has_read_criteria(self):
        plan = build_contract_plan_trace(make_input(
            intent=IntentContract(
                kind="research",
                tool_requirement="read",
                grounding_requirement="none",
                output_expectation="text",
                requires_code_verification=False,
            ),
        ))
        assert any("read" in c.lower() for c in plan.completion_criteria)


# ---------------------------------------------------------------------------
# build_probe_candidates
# ---------------------------------------------------------------------------

class TestBuildProbeCandidates:
    def test_go_probe(self):
        probes = build_probe_candidates(["go.mod"])
        assert len(probes) >= 1
        assert any("go" in p.lower() for p in probes)

    def test_node_probe(self):
        probes = build_probe_candidates(["package.json"])
        assert len(probes) >= 1
        assert any("node" in p.lower() for p in probes)

    def test_python_probe(self):
        probes = build_probe_candidates(["pyproject.toml"])
        assert len(probes) >= 1
        assert any("python" in p.lower() for p in probes)

    def test_empty_workspace_files(self):
        probes = build_probe_candidates([])
        assert len(probes) == 0
