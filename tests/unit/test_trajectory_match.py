"""Tests for rune.evaluation.metrics.trajectory_match — trajectory matching and scoring."""

from __future__ import annotations

from rune.evaluation.metrics.trajectory_match import (
    TrajectoryStep,
    calculate_trajectory_score,
    detect_repetition,
    get_tool_frequency,
    has_sequence,
    match_trajectory,
    tool_called_before,
)


def _step(tool: str) -> TrajectoryStep:
    return TrajectoryStep(tool=tool)


# ---------------------------------------------------------------------------
# match_trajectory — strict mode
# ---------------------------------------------------------------------------


class TestMatchTrajectoryStrict:
    def test_identical_trajectories(self):
        actual = [_step("file.read"), _step("bash")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is True

    def test_different_order_fails(self):
        actual = [_step("bash"), _step("file.read")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False

    def test_extra_tools_fail(self):
        actual = [_step("file.read"), _step("bash"), _step("file.write")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False


# ---------------------------------------------------------------------------
# match_trajectory — superset mode
# ---------------------------------------------------------------------------


class TestMatchTrajectorySuperSet:
    def test_passes_when_all_expected_present(self):
        actual = [_step("file.read"), _step("bash"), _step("file.write")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "superset")
        assert result.matched is True

    def test_fails_when_expected_missing(self):
        actual = [_step("file.read")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "superset")
        assert result.matched is False
        assert "bash" in result.missing_tools


# ---------------------------------------------------------------------------
# match_trajectory — subset mode
# ---------------------------------------------------------------------------


class TestMatchTrajectorySubset:
    def test_passes_when_only_expected_tools_used(self):
        actual = [_step("file.read")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "subset")
        assert result.matched is True

    def test_fails_when_extra_tools_used(self):
        actual = [_step("file.read"), _step("browser")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "subset")
        assert result.matched is False
        assert "browser" in result.extra_tools


# ---------------------------------------------------------------------------
# match_trajectory — unordered mode
# ---------------------------------------------------------------------------


class TestMatchTrajectoryUnordered:
    def test_same_tools_different_order(self):
        actual = [_step("bash"), _step("file.read")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_fails_when_sets_differ(self):
        actual = [_step("file.read"), _step("browser")]
        expected = [_step("file.read"), _step("bash")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False


# ---------------------------------------------------------------------------
# calculate_trajectory_score
# ---------------------------------------------------------------------------


class TestCalculateTrajectoryScore:
    def test_perfect_match_returns_1(self):
        result = match_trajectory([_step("file.read")], [_step("file.read")], "strict")
        assert calculate_trajectory_score(result) == 1.0

    def test_missing_tools_penalized(self):
        result = match_trajectory(
            [_step("file.read")],
            [_step("file.read"), _step("bash")],
            "superset",
        )
        score = calculate_trajectory_score(result)
        assert score < 1.0

    def test_extra_tools_penalized_less_than_missing(self):
        result_missing = match_trajectory([], [_step("bash")], "superset")
        result_extra = match_trajectory(
            [_step("file.read"), _step("bash")],
            [_step("file.read")],
            "subset",
        )
        assert calculate_trajectory_score(result_missing) < calculate_trajectory_score(result_extra)


# ---------------------------------------------------------------------------
# has_sequence
# ---------------------------------------------------------------------------


class TestHasSequence:
    def test_detects_subsequence(self):
        trajectory = [_step("file.read"), _step("bash"), _step("file.write")]
        assert has_sequence(trajectory, ["file.read", "file.write"]) is True
        assert has_sequence(trajectory, ["file.read", "bash"]) is True

    def test_returns_false_for_missing(self):
        trajectory = [_step("file.read"), _step("bash")]
        assert has_sequence(trajectory, ["file.write", "bash"]) is False

    def test_empty_sequence(self):
        assert has_sequence([_step("bash")], []) is True


# ---------------------------------------------------------------------------
# tool_called_before
# ---------------------------------------------------------------------------


class TestToolCalledBefore:
    def test_detects_ordering(self):
        trajectory = [_step("file.read"), _step("bash"), _step("file.write")]
        assert tool_called_before(trajectory, "file.read", "bash") is True
        assert tool_called_before(trajectory, "bash", "file.read") is False

    def test_returns_false_when_tools_not_present(self):
        trajectory = [_step("file.read")]
        assert tool_called_before(trajectory, "file.read", "bash") is False


# ---------------------------------------------------------------------------
# get_tool_frequency
# ---------------------------------------------------------------------------


class TestGetToolFrequency:
    def test_counts_occurrences(self):
        trajectory = [_step("bash"), _step("file.read"), _step("bash"), _step("bash")]
        freq = get_tool_frequency(trajectory)
        assert freq["bash"] == 3
        assert freq["file.read"] == 1


# ---------------------------------------------------------------------------
# detect_repetition
# ---------------------------------------------------------------------------


class TestDetectRepetition:
    def test_detects_repeated_tools(self):
        trajectory = [_step("bash")] * 5
        result = detect_repetition(trajectory, threshold=3)
        assert result["detected"] is True
        assert "bash" in result["repeated_tools"]

    def test_no_repetition(self):
        trajectory = [_step("bash"), _step("file.read")]
        result = detect_repetition(trajectory, threshold=3)
        assert result["detected"] is False
