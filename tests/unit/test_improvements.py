"""Tests for the three improvements: B (R06 verifications), C (edit-loop), D (cycle guidance)."""
import pytest
from rune.agent.loop import StallState
from rune.agent.tool_adapter import _update_stall_state, STALL_LIMITS
from rune.agent.completion_gate import ExecutionEvidenceSnapshot, evaluate_completion_gate, CompletionGateInput


class _FakeResult:
    def __init__(self, success: bool = True, error: str = ""):
        self.success = success
        self.error = error


# ── B: R06 verifications counter ──


class TestR06VerificationsCounter:
    def test_verification_command_increments(self):
        """is_verification_command-matching bash should increment evidence.verifications."""
        from rune.agent.bash_parsing import is_verification_command
        assert is_verification_command("pytest tests/ -v")
        assert is_verification_command("ruff check .")
        assert not is_verification_command("echo hello")
        assert not is_verification_command("python3 script.py")

    def test_r06_passes_with_verifications(self):
        """R06 should pass when verifications > 0."""
        ev = ExecutionEvidenceSnapshot(reads=1, writes=1, executions=1, verifications=1)
        inp = CompletionGateInput(
            intent_resolved=True,
            tool_requirement="write",
            output_expectation="file",
            evidence=ev,
            changed_files_count=1,
            answer_length=100,
            requires_code_verification=True,
        )
        result = evaluate_completion_gate(inp)
        r06 = [r for r in result.requirements if "VERIFICATION" in r.id]
        assert len(r06) == 1
        assert r06[0].status == "done"

    def test_r06_blocked_without_verifications(self):
        """R06 should be blocked when verifications == 0 and code verification required."""
        ev = ExecutionEvidenceSnapshot(reads=1, writes=1, executions=1, verifications=0)
        inp = CompletionGateInput(
            intent_resolved=True,
            tool_requirement="write",
            output_expectation="file",
            evidence=ev,
            changed_files_count=1,
            answer_length=100,
            requires_code_verification=True,
        )
        result = evaluate_completion_gate(inp)
        r06 = [r for r in result.requirements if "VERIFICATION" in r.id]
        assert len(r06) == 1
        assert r06[0].status == "blocked"


# ── C: Edit-loop circuit breaker ──


class TestEditLoopBreaker:
    def test_file_edit_counts_tracking(self):
        """file_edit_counts should track per-file edit counts."""
        stall = StallState()
        fp = "/tmp/test.py"

        for i in range(5):
            _update_stall_state(
                stall, "file_edit",
                {"file_path": fp},
                _FakeResult(success=True),
                100.0,
            )

        assert stall.file_edit_counts[fp] == 5

    def test_stall_warning_at_threshold(self):
        """stall_warning_issued should be set when same file edited >= sameFile limit."""
        stall = StallState()
        fp = "/tmp/test.py"
        limit = STALL_LIMITS["fileWrite"]["sameFile"]  # 3

        for i in range(limit - 1):
            _update_stall_state(
                stall, "file_edit", {"file_path": fp},
                _FakeResult(success=True), 100.0,
            )
        assert not stall.stall_warning_issued

        _update_stall_state(
            stall, "file_edit", {"file_path": fp},
            _FakeResult(success=True), 100.0,
        )
        assert stall.stall_warning_issued

    def test_different_files_tracked_separately(self):
        """Different file paths should have independent counters."""
        stall = StallState()
        for i in range(3):
            _update_stall_state(
                stall, "file_edit", {"file_path": "/tmp/a.py"},
                _FakeResult(success=True), 100.0,
            )
        _update_stall_state(
            stall, "file_edit", {"file_path": "/tmp/b.py"},
            _FakeResult(success=True), 100.0,
        )
        assert stall.file_edit_counts["/tmp/a.py"] == 3
        assert stall.file_edit_counts["/tmp/b.py"] == 1

    def test_file_write_also_tracked(self):
        """file_write (not just file_edit) should also be tracked."""
        stall = StallState()
        fp = "/tmp/test.py"
        limit = STALL_LIMITS["fileWrite"]["sameFile"]
        for i in range(limit):
            _update_stall_state(
                stall, "file_write", {"file_path": fp},
                _FakeResult(success=True), 100.0,
            )
        assert stall.file_edit_counts[fp] == limit
        assert stall.stall_warning_issued

    def test_hard_block_threshold(self):
        """After sameFile edits, file_edit_counts should exceed the limit."""
        stall = StallState()
        fp = "/tmp/loop.py"
        limit = STALL_LIMITS["fileWrite"]["sameFile"]
        for i in range(limit + 2):
            _update_stall_state(
                stall, "file_edit", {"file_path": fp},
                _FakeResult(success=True), 100.0,
            )
        assert stall.file_edit_counts[fp] == limit + 2
        assert stall.file_edit_counts[fp] >= limit


# ── D: Cycle detection guidance ──


class TestCycleGuidance:
    def test_cycle_detected_after_repeating_pattern(self):
        """2-step repeating pattern should set cycle_detected."""
        stall = StallState()
        for _ in range(3):
            _update_stall_state(
                stall, "file_edit", {"file_path": "/tmp/t.py"},
                _FakeResult(success=True), 100.0,
            )
            _update_stall_state(
                stall, "bash_execute", {"command": "python3 /tmp/t.py"},
                _FakeResult(success=True), 100.0,
            )
        assert stall.cycle_detected

    def test_no_cycle_for_varied_calls(self):
        """Non-repeating tool calls should not trigger cycle detection."""
        stall = StallState()
        tools = ["file_read", "file_edit", "bash_execute", "web_search", "file_write"]
        for t in tools:
            _update_stall_state(
                stall, t, {}, _FakeResult(success=True), 100.0,
            )
        assert not stall.cycle_detected
