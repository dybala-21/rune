"""Tests for rune.agent.plan_validator — static plan validation."""


from rune.agent.plan_validator import (
    SubTask,
    detect_cycle,
    validate_plan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(id: str, **kwargs) -> SubTask:
    return SubTask(
        id=id,
        goal=kwargs.get("goal", f"do {id}"),
        role=kwargs.get("role", "executor"),
        depends_on=kwargs.get("depends_on", []),
    )


# ---------------------------------------------------------------------------
# detect_cycle
# ---------------------------------------------------------------------------

class TestDetectCycle:
    def test_detects_direct_cycle(self):
        tasks = [
            make_task("t1", depends_on=["t2"]),
            make_task("t2", depends_on=["t1"]),
        ]
        cycle = detect_cycle(tasks)
        assert cycle is not None
        assert len(cycle) > 1

    def test_detects_indirect_cycle(self):
        tasks = [
            make_task("t1", depends_on=["t3"]),
            make_task("t2", depends_on=["t1"]),
            make_task("t3", depends_on=["t2"]),
        ]
        cycle = detect_cycle(tasks)
        assert cycle is not None

    def test_returns_none_for_valid_dag(self):
        tasks = [
            make_task("t1"),
            make_task("t2", depends_on=["t1"]),
            make_task("t3", depends_on=["t1", "t2"]),
        ]
        assert detect_cycle(tasks) is None

    def test_returns_none_for_empty_tasks(self):
        assert detect_cycle([]) is None

    def test_returns_none_for_single_task(self):
        assert detect_cycle([make_task("t1")]) is None

    def test_self_cycle(self):
        tasks = [make_task("t1", depends_on=["t1"])]
        cycle = detect_cycle(tasks)
        assert cycle is not None


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_approves_valid_plan(self):
        tasks = [
            make_task("t1", goal="search for hotels"),
            make_task("t2", goal="summarize results", depends_on=["t1"]),
        ]
        result = validate_plan(tasks, "find hotels and summarize")
        assert result.approved is True

    def test_rejects_plan_with_cycle(self):
        tasks = [
            make_task("t1", depends_on=["t2"]),
            make_task("t2", depends_on=["t1"]),
        ]
        result = validate_plan(tasks, "test")
        assert result.approved is False
        assert any(i.type == "cycle" for i in result.issues)

    def test_rejects_plan_with_dangling_dependency(self):
        tasks = [make_task("t1", depends_on=["nonexistent"])]
        result = validate_plan(tasks, "test")
        assert result.approved is False
        assert any(i.type == "dangling_dep" for i in result.issues)

    def test_warns_about_over_decomposition(self):
        # Goal has ~3 meaningful words but plan has 12 tasks
        tasks = [make_task(f"t{i}") for i in range(12)]
        result = validate_plan(tasks, "find three hotels")
        assert result.approved is True  # warnings don't block
        assert any(i.type == "over_decomposition" for i in result.issues)

    def test_warns_about_resource_waste(self):
        # 10 executor tasks = 10 * 5min = 50min >> 3 * 5min = 15min threshold
        tasks = [make_task(f"t{i}", role="executor") for i in range(10)]
        result = validate_plan(tasks, "big task")
        assert any(i.type == "resource_waste" for i in result.issues)

    def test_no_role_mismatch_for_executor_default(self):
        tasks = [make_task("t1", goal="some generic task", role="executor")]
        result = validate_plan(tasks, "test")
        role_mismatches = [i for i in result.issues if i.type == "role_mismatch"]
        assert len(role_mismatches) == 0

    def test_suggestion_contains_error_messages(self):
        tasks = [
            make_task("t1", depends_on=["t2"]),
            make_task("t2", depends_on=["t1"]),
        ]
        result = validate_plan(tasks, "test")
        assert result.suggestion is not None
        assert "Cyclic" in result.suggestion

    def test_empty_plan_approved(self):
        result = validate_plan([], "test")
        assert result.approved is True
