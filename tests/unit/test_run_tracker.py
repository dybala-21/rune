"""Tests for rune.api.run_tracker — ported from run-tracker.test.ts."""


from rune.api.run_tracker import RunResult, RunTracker, TokenUsage


class TestRunTrackerCreate:
    """Tests for RunTracker.create()."""

    def test_creates_run_with_queued_status(self):
        tracker = RunTracker()
        state = tracker.create("r1", "c1", "s1", "do stuff")
        assert state.run_id == "r1"
        assert state.session_id == "s1"
        assert state.client_id == "c1"
        assert state.status == "queued"
        assert state.goal == "do stuff"
        assert state.started_at > 0

    def test_stores_run_retrievable_by_get(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        run = tracker.get("r1")
        assert run is not None
        assert run.run_id == "r1"

    def test_increments_active_count(self):
        tracker = RunTracker()
        assert tracker.get_active_count() == 0
        tracker.create("r1", "c1", "s1", "goal")
        assert tracker.get_active_count() == 1
        tracker.create("r2", "c1", "s1", "goal2")
        assert tracker.get_active_count() == 2


class TestRunTrackerMarkRunning:
    """Tests for RunTracker.mark_running()."""

    def test_transitions_queued_run_to_running(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_running("r1")
        assert tracker.get("r1").status == "running"

    def test_does_nothing_for_nonexistent_run_id(self):
        tracker = RunTracker()
        tracker.mark_running("nonexistent")  # should not raise


class TestRunTrackerMarkCompleted:
    """Tests for RunTracker.mark_completed()."""

    def test_transitions_to_completed_with_result(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        result = RunResult(success=True, answer="done", usage=TokenUsage(input=100, output=50))
        tracker.mark_completed("r1", result)

        run = tracker.get("r1")
        assert run.status == "completed"
        assert run.completed_at is not None
        assert run.result == result

    def test_decreases_active_count(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        assert tracker.get_active_count() == 1
        tracker.mark_completed("r1", RunResult(success=True, answer="ok"))
        assert tracker.get_active_count() == 0
        # But still retrievable
        assert tracker.get("r1") is not None

    def test_does_nothing_for_nonexistent_run_id(self):
        tracker = RunTracker()
        tracker.mark_completed("nonexistent", RunResult(success=True, answer="ok"))
        assert tracker.get("nonexistent") is None


class TestRunTrackerMarkFailed:
    """Tests for RunTracker.mark_failed()."""

    def test_transitions_to_failed_with_error(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_failed("r1", "something broke")

        run = tracker.get("r1")
        assert run.status == "failed"
        assert run.completed_at is not None
        assert run.error == "something broke"

    def test_moves_run_from_active_to_completed(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_failed("r1", "err")
        assert tracker.get_active_count() == 0
        assert tracker.get("r1") is not None

    def test_does_nothing_for_nonexistent_run_id(self):
        tracker = RunTracker()
        tracker.mark_failed("nonexistent", "err")
        assert tracker.get("nonexistent") is None


class TestRunTrackerMarkAborted:
    """Tests for RunTracker.mark_aborted()."""

    def test_transitions_to_aborted(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_aborted("r1")

        run = tracker.get("r1")
        assert run.status == "aborted"
        assert run.completed_at is not None

    def test_moves_run_from_active_to_completed(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_aborted("r1")
        assert tracker.get_active_count() == 0
        assert tracker.get("r1") is not None

    def test_does_nothing_for_nonexistent_run_id(self):
        tracker = RunTracker()
        tracker.mark_aborted("nonexistent")
        assert tracker.get("nonexistent") is None


class TestRunTrackerGet:
    """Tests for RunTracker.get()."""

    def test_returns_none_for_unknown_run_id(self):
        tracker = RunTracker()
        assert tracker.get("unknown") is None

    def test_finds_active_runs(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        assert tracker.get("r1").status == "queued"

    def test_finds_completed_runs(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        tracker.mark_completed("r1", RunResult(success=True, answer="done"))
        assert tracker.get("r1").status == "completed"


class TestRunTrackerGetForClient:
    """Tests for RunTracker.get_for_client()."""

    def test_returns_empty_when_no_matching_client(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal")
        assert tracker.get_for_client("c2") == []

    def test_returns_all_runs_for_client(self):
        tracker = RunTracker()
        tracker.create("r1", "c1", "s1", "goal1")
        tracker.create("r2", "c1", "s1", "goal2")
        tracker.create("r3", "c2", "s1", "goal3")
        runs = tracker.get_for_client("c1")
        assert len(runs) == 2
        assert {r.run_id for r in runs} == {"r1", "r2"}
