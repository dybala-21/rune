"""Episode adapter for RUNE evaluation system.

Converts memory Episode objects into evaluation TestCase structures and
trajectory steps.  Supports filtering, batch conversion, and export to
DeepEval-compatible JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class EvalTestCase:
    """An evaluation test case derived from a memory episode."""

    id: str = ""
    input: str = ""
    actual_output: str = ""
    expected_output: str = ""
    context: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    expected_tools: list[str] | None = None
    is_negative_test: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryStep:
    """A tool-call step extracted from an episode plan."""

    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)


# Episode -> TestCase / Trajectory

def episode_to_test_case(episode: Any) -> EvalTestCase:
    """Convert a memory Episode to an :class:`EvalTestCase`.

    Parameters:
        episode: An Episode-like object with ``id``, ``task_summary`` (or
            ``taskSummary``), ``result``, ``intent``, ``lessons``, and
            ``plan`` attributes.

    Returns:
        A populated EvalTestCase.
    """
    task_summary = getattr(episode, "task_summary", None) or getattr(episode, "taskSummary", "")
    result = getattr(episode, "result", None) or {}
    intent = getattr(episode, "intent", None)
    lessons = getattr(episode, "lessons", []) or []

    answer = (
        getattr(result, "answer", None)
        or getattr(result, "summary", None)
        or (result.get("answer") if isinstance(result, dict) else "")
        or (result.get("summary") if isinstance(result, dict) else "")
        or ""
    )

    iterations = (
        getattr(result, "iterations", None)
        or getattr(result, "steps_completed", None)
        or (result.get("iterations") if isinstance(result, dict) else 0)
        or (result.get("stepsCompleted") if isinstance(result, dict) else 0)
        or 0
    )

    success = (
        getattr(result, "success", False)
        if not isinstance(result, dict)
        else result.get("success", False)
    )

    action = ""
    domain = ""
    if intent is not None:
        action = getattr(intent, "action", "") or (intent.get("action", "") if isinstance(intent, dict) else "")
        domain = getattr(intent, "domain", "") or (intent.get("domain", "") if isinstance(intent, dict) else "")

    timestamp = getattr(episode, "timestamp", None)

    return EvalTestCase(
        id=getattr(episode, "id", ""),
        input=task_summary,
        actual_output=answer,
        expected_output=action,
        context=list(lessons) if lessons else [],
        tools_called=extract_tools_from_plan(episode),
        metadata={
            "success": success,
            "iterations": iterations,
            "timestamp": str(timestamp) if timestamp else "",
            "domain": domain,
        },
    )


def episode_to_trajectory(episode: Any) -> list[TrajectoryStep]:
    """Extract tool-call trajectory from an episode's plan."""
    plan = getattr(episode, "plan", None)
    if plan is None:
        return []
    steps = getattr(plan, "steps", []) or []
    trajectory: list[TrajectoryStep] = []
    for step in steps:
        tool = getattr(step, "tool", None) or (step.get("tool") if isinstance(step, dict) else None)
        if tool:
            args = getattr(step, "params", {}) or (step.get("params", {}) if isinstance(step, dict) else {})
            trajectory.append(TrajectoryStep(tool=tool, args=dict(args) if args else {}))
    return trajectory


def extract_tools_from_plan(episode: Any) -> list[str]:
    """Extract the list of tool names from an episode's plan."""
    return [s.tool for s in episode_to_trajectory(episode)]


# Batch operations

def episodes_to_test_cases(episodes: list[Any]) -> list[EvalTestCase]:
    """Batch convert episodes to test cases."""
    return [episode_to_test_case(ep) for ep in episodes]


def export_for_deepeval(
    test_cases: list[EvalTestCase],
    *,
    pretty: bool = False,
) -> str:
    """Export test cases as DeepEval-compatible JSON.

    Parameters:
        test_cases: Test cases to export.
        pretty: If ``True``, indent the JSON for readability.

    Returns:
        JSON string.
    """
    formatted = [
        {
            "id": tc.id,
            "input": tc.input,
            "actual_output": tc.actual_output,
            "expected_output": tc.expected_output,
            "context": tc.context,
            "tools_called": tc.tools_called,
            "expected_tools": tc.expected_tools,
            "is_negative_test": tc.is_negative_test,
            "metadata": tc.metadata,
        }
        for tc in test_cases
    ]
    return json.dumps(formatted, indent=2 if pretty else None)


# Filtering

def filter_episodes_for_eval(
    episodes: list[Any],
    *,
    only_success: bool = False,
    only_failure: bool = False,
    domains: list[str] | None = None,
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    limit: int | None = None,
) -> list[Any]:
    """Filter episodes for evaluation.

    Parameters:
        episodes: Source episodes.
        only_success: Keep only successful episodes.
        only_failure: Keep only failed episodes.
        domains: Allowed intent domains.
        min_date: Earliest timestamp.
        max_date: Latest timestamp.
        limit: Maximum number of episodes to return (most recent first).

    Returns:
        Filtered (and optionally limited) list of episodes.
    """
    filtered = list(episodes)

    if only_success:
        filtered = [
            ep for ep in filtered
            if _episode_success(ep)
        ]

    if only_failure:
        filtered = [
            ep for ep in filtered
            if not _episode_success(ep)
        ]

    if domains:
        filtered = [
            ep for ep in filtered
            if _episode_domain(ep) in domains
        ]

    if min_date is not None:
        filtered = [
            ep for ep in filtered
            if _episode_timestamp(ep) is not None and _episode_timestamp(ep) >= min_date  # type: ignore[operator]
        ]

    if max_date is not None:
        filtered = [
            ep for ep in filtered
            if _episode_timestamp(ep) is not None and _episode_timestamp(ep) <= max_date  # type: ignore[operator]
        ]

    # Sort by timestamp descending (most recent first)
    filtered.sort(
        key=lambda ep: _episode_timestamp(ep) or datetime.min,
        reverse=True,
    )

    if limit is not None:
        filtered = filtered[:limit]

    return filtered


# Internal helpers

def _episode_success(episode: Any) -> bool:
    result = getattr(episode, "result", None)
    if result is None:
        return False
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(getattr(result, "success", False))


def _episode_domain(episode: Any) -> str:
    intent = getattr(episode, "intent", None)
    if intent is None:
        return ""
    if isinstance(intent, dict):
        return intent.get("domain", "")
    return getattr(intent, "domain", "")


def _episode_timestamp(episode: Any) -> datetime | None:
    ts = getattr(episode, "timestamp", None)
    if isinstance(ts, datetime):
        return ts
    return None
