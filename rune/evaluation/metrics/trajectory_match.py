"""Trajectory matching for RUNE evaluation system.

Compares actual agent tool-call trajectories against expected trajectories
using multiple matching strategies (strict, superset, subset, unordered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TrajectoryMatchMode = Literal["strict", "superset", "subset", "unordered"]


@dataclass(slots=True)
class TrajectoryStep:
    """A single step in an agent trajectory."""

    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryMatchResult:
    """Result of comparing two trajectories."""

    matched: bool = False
    mode: TrajectoryMatchMode = "superset"
    expected_steps: int = 0
    actual_steps: int = 0
    matched_steps: int = 0
    missing_tools: list[str] = field(default_factory=list)
    extra_tools: list[str] = field(default_factory=list)


# Core matching

def match_trajectory(
    actual: list[TrajectoryStep],
    expected: list[TrajectoryStep],
    mode: TrajectoryMatchMode = "superset",
) -> TrajectoryMatchResult:
    """Match *actual* trajectory against *expected*.

    Parameters:
        actual: The observed tool-call sequence.
        expected: The desired tool-call sequence.
        mode: Matching strategy - ``strict`` (exact order), ``superset``
            (actual includes all expected), ``subset`` (actual only uses
            expected tools), or ``unordered`` (same set, any order).

    Returns:
        A :class:`TrajectoryMatchResult` with match details.
    """
    actual_tools = [s.tool for s in actual]
    expected_tools = [s.tool for s in expected]

    result = TrajectoryMatchResult(
        mode=mode,
        expected_steps=len(expected),
        actual_steps=len(actual),
    )

    if mode == "strict":
        result.matched = actual_tools == expected_tools
        if not result.matched:
            result.missing_tools = [
                t for i, t in enumerate(expected_tools)
                if i >= len(actual_tools) or actual_tools[i] != t
            ]
            result.extra_tools = [
                t for i, t in enumerate(actual_tools)
                if i >= len(expected_tools) or expected_tools[i] != t
            ]

    elif mode == "superset":
        result.missing_tools = [
            t for t in expected_tools if t not in actual_tools
        ]
        result.matched = len(result.missing_tools) == 0

    elif mode == "subset":
        result.extra_tools = [
            t for t in actual_tools if t not in expected_tools
        ]
        result.matched = len(result.extra_tools) == 0

    elif mode == "unordered":
        actual_set = set(actual_tools)
        expected_set = set(expected_tools)
        result.matched = actual_set == expected_set
        if not result.matched:
            result.missing_tools = [t for t in expected_set if t not in actual_set]
            result.extra_tools = [t for t in actual_set if t not in expected_set]

    result.matched_steps = sum(1 for t in actual_tools if t in expected_tools)
    return result


# Scoring

def calculate_trajectory_score(result: TrajectoryMatchResult) -> float:
    """Calculate a 0-1 score from a trajectory match result.

    Penalises missing tools heavily (0.2 each) and extra tools lightly
    (0.05 each).
    """
    if result.matched:
        return 1.0
    if result.expected_steps == 0:
        return 1.0 if result.actual_steps == 0 else 0.5

    match_ratio = result.matched_steps / result.expected_steps
    missing_penalty = len(result.missing_tools) * 0.2
    extra_penalty = len(result.extra_tools) * 0.05
    return max(0.0, min(1.0, match_ratio - missing_penalty - extra_penalty))


# Sequence helpers

def has_sequence(trajectory: list[TrajectoryStep], sequence: list[str]) -> bool:
    """Return ``True`` if *sequence* appears as a subsequence of *trajectory*.

    The subsequence does not need to be contiguous.
    """
    if not sequence:
        return True
    if len(trajectory) < len(sequence):
        return False

    tools = [s.tool for s in trajectory]
    seq_idx = 0
    for tool in tools:
        if tool == sequence[seq_idx]:
            seq_idx += 1
            if seq_idx == len(sequence):
                return True
    return False


def tool_called_before(
    trajectory: list[TrajectoryStep],
    first_tool: str,
    second_tool: str,
) -> bool:
    """Return ``True`` if *first_tool* appears before *second_tool*."""
    tools = [s.tool for s in trajectory]
    try:
        return tools.index(first_tool) < tools.index(second_tool)
    except ValueError:
        return False


def get_tool_frequency(trajectory: list[TrajectoryStep]) -> dict[str, int]:
    """Return a mapping of tool name to call count."""
    freq: dict[str, int] = {}
    for step in trajectory:
        freq[step.tool] = freq.get(step.tool, 0) + 1
    return freq


def detect_repetition(
    trajectory: list[TrajectoryStep],
    threshold: int = 3,
) -> dict[str, Any]:
    """Detect tools called >= *threshold* times (potential loop).

    Returns:
        Dictionary with ``detected`` (bool) and ``repeated_tools`` (list).
    """
    freq = get_tool_frequency(trajectory)
    repeated = [t for t, c in freq.items() if c >= threshold]
    return {"detected": len(repeated) > 0, "repeated_tools": repeated}
