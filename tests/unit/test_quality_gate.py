"""Tests for rune.agent.quality_gate — sub-agent result quality verification."""


from rune.agent.quality_gate import (
    AgentResult,
    TaskInfo,
    check_task_quality,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(role: str = "researcher") -> TaskInfo:
    return TaskInfo(id="t1", role=role, goal="do something")


def make_result(**kwargs) -> AgentResult:
    defaults = dict(
        success=True,
        answer="This is a detailed and comprehensive result with sufficient length to pass quality checks.",
        iterations=5,
        duration_ms=5000.0,
        history=[],
    )
    defaults.update(kwargs)
    return AgentResult(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_passes_good_results(self):
        check = check_task_quality(make_task(), make_result())
        assert check.passed is True
        assert check.score == 1.0
        assert len(check.issues) == 0

    def test_always_passes_failed_results(self):
        check = check_task_quality(
            make_task(),
            make_result(success=False, answer=""),
        )
        assert check.passed is True

    def test_fails_on_very_short_answers(self):
        check = check_task_quality(
            make_task(),
            make_result(answer="ok"),
        )
        assert check.passed is False
        assert check.score <= 0.2
        assert any("short" in i.lower() or "chars" in i.lower() for i in check.issues)

    def test_passes_on_sufficiently_long_answers(self):
        check = check_task_quality(
            make_task(),
            make_result(answer="A" * 60),
        )
        hollow_issues = [i for i in check.issues if "short" in i.lower() or "chars" in i.lower()]
        assert len(hollow_issues) == 0

    def test_fails_executor_with_no_action_evidence(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(iterations=1, history=[]),
        )
        assert check.passed is False
        assert any("evidence" in i.lower() or "action" in i.lower() for i in check.issues)

    def test_passes_executor_with_sufficient_iterations(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(iterations=5),
        )
        evidence_issues = [i for i in check.issues if "evidence" in i.lower() and "action" in i.lower()]
        assert len(evidence_issues) == 0

    def test_passes_executor_with_action_history(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(
                iterations=2,
                history=[{"type": "action", "content": "file.write"}],
            ),
        )
        evidence_issues = [i for i in check.issues if "evidence" in i.lower() and "action" in i.lower()]
        assert len(evidence_issues) == 0

    def test_no_execution_evidence_check_for_researcher(self):
        check = check_task_quality(
            make_task("researcher"),
            make_result(iterations=1, history=[]),
        )
        evidence_issues = [i for i in check.issues if "Executor" in i]
        assert len(evidence_issues) == 0

    def test_warns_on_multiple_error_keywords(self):
        check = check_task_quality(
            make_task(),
            make_result(
                answer="I could not find the file. Failed to access the API. Unable to complete the task properly.",
            ),
        )
        assert any("error" in i.lower() or "keyword" in i.lower() for i in check.issues)

    def test_no_warn_on_single_error_keyword(self):
        check = check_task_quality(
            make_task(),
            make_result(
                answer="The search found results. One error was logged in the server but it was non-critical and handled properly in the code.",
            ),
        )
        # Only 1 error keyword — not suspicious (threshold is >= 2)
        masking_issues = [i for i in check.issues if "error" in i.lower() and "keyword" in i.lower()]
        assert len(masking_issues) == 0

    def test_warns_when_executor_completes_too_fast(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(duration_ms=500.0, iterations=5),
        )
        assert any("500" in i or "ms" in i.lower() for i in check.issues)

    def test_no_speed_warn_for_communicator(self):
        check = check_task_quality(
            make_task("communicator"),
            make_result(duration_ms=500.0),
        )
        speed_issues = [i for i in check.issues if "ms" in i.lower() and "completed" in i.lower()]
        assert len(speed_issues) == 0

    def test_no_speed_warn_for_normal_executor(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(duration_ms=10000.0, iterations=5),
        )
        speed_issues = [i for i in check.issues if "may not" in i.lower()]
        assert len(speed_issues) == 0

    def test_suggestion_present_when_failed(self):
        check = check_task_quality(
            make_task("executor"),
            make_result(answer="ok", iterations=1, history=[]),
        )
        assert check.passed is False
        assert check.suggestion is not None
        assert "retry" in check.suggestion.lower() or "execute" in check.suggestion.lower()
