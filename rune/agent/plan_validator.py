"""Plan validator - static plan validation before execution.

Ported from src/agent/plan-validator.ts (163 lines) - zero LLM cost,
sub-millisecond static analysis of orchestration plans.

Checks:
- Cyclic dependency detection (DFS)
- Dangling dependency references
- Role-goal mismatch
- Over-decomposition
- Resource waste warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

PlanIssueSeverity = Literal["error", "warning"]
PlanIssueType = Literal[
    "cycle",
    "role_mismatch",
    "over_decomposition",
    "dangling_dep",
    "resource_waste",
]


@dataclass(slots=True)
class PlanIssue:
    """A single issue detected in a plan."""

    severity: PlanIssueSeverity
    type: PlanIssueType
    message: str
    task_id: str | None = None


@dataclass(slots=True)
class PlanValidation:
    """Result of static plan validation."""

    approved: bool
    issues: list[PlanIssue] = field(default_factory=list)
    suggestion: str | None = None


@dataclass(slots=True)
class SubTask:
    """Minimal sub-task representation for validation."""

    id: str
    goal: str = ""
    role: str = "executor"
    depends_on: list[str] = field(default_factory=list)


# Default timeout (executor role baseline)

_SINGLE_AGENT_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes default


def _get_role_timeout(role: str) -> int:
    """Return timeout in ms for a given role name."""
    try:
        from rune.agent.roles import get_role

        r = get_role(role)
        return getattr(r, "timeout", _SINGLE_AGENT_TIMEOUT_MS)
    except Exception:
        return _SINGLE_AGENT_TIMEOUT_MS


# Cycle detection (DFS)

def detect_cycle(tasks: list[SubTask]) -> list[str] | None:
    """Detect cyclic dependencies in task graph using DFS.

    Returns the cycle path (list of task IDs) if found, otherwise None.
    """
    task_map: dict[str, SubTask] = {t.id: t for t in tasks}
    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []

    def dfs(task_id: str) -> list[str] | None:
        if task_id in in_stack:
            cycle_start = path.index(task_id)
            return path[cycle_start:] + [task_id]
        if task_id in visited:
            return None

        visited.add(task_id)
        in_stack.add(task_id)
        path.append(task_id)

        task = task_map.get(task_id)
        if task:
            for dep in task.depends_on:
                result = dfs(dep)
                if result is not None:
                    return result

        path.pop()
        in_stack.discard(task_id)
        return None

    for t in tasks:
        result = dfs(t.id)
        if result is not None:
            return result

    return None


# Role suggestion (lightweight sync)

def _suggest_role(goal: str) -> str:
    """Suggest a role for a goal string. Returns 'executor' as safe default."""
    try:
        from rune.agent.roles import suggest_role_with_intent_sync

        return suggest_role_with_intent_sync(goal)
    except Exception:
        return "executor"


# validate_plan

def validate_plan(
    tasks: list[SubTask],
    original_goal: str,
) -> PlanValidation:
    """Static plan validation - zero LLM cost, <1ms.

    Checks:
    1. Cyclic dependencies
    2. Dangling dependency references
    3. Role-goal mismatch
    4. Over-decomposition
    5. Resource waste
    """
    issues: list[PlanIssue] = []

    # 1. Cycle detection
    cycle = detect_cycle(tasks)
    if cycle:
        issues.append(PlanIssue(
            severity="error",
            type="cycle",
            message=f"Cyclic dependency: {' -> '.join(cycle)}",
        ))

    # 2. Dangling dependency references
    task_ids = {t.id for t in tasks}
    for task in tasks:
        for dep in task.depends_on:
            if dep not in task_ids:
                issues.append(PlanIssue(
                    severity="error",
                    type="dangling_dep",
                    task_id=task.id,
                    message=f'Task "{task.id}" depends on non-existent "{dep}"',
                ))

    # 3. Role-goal mismatch
    for task in tasks:
        suggested = _suggest_role(task.goal)
        if suggested != task.role and task.role != "executor":
            issues.append(PlanIssue(
                severity="warning",
                type="role_mismatch",
                task_id=task.id,
                message=(
                    f'Task "{task.id}" role "{task.role}" mismatches goal '
                    f'(suggested: "{suggested}")'
                ),
            ))

    # 4. Over-decomposition
    import re

    goal_words = [w for w in re.split(r"\s+", re.sub(r"[^\w\s]", "", original_goal)) if len(w) > 1]
    word_count = max(len(goal_words), 3)
    if len(tasks) > word_count * 2:
        issues.append(PlanIssue(
            severity="warning",
            type="over_decomposition",
            message=(
                f"{len(tasks)} tasks is over-decomposed for goal complexity "
                f"({word_count} words)"
            ),
        ))

    # 5. Resource waste
    total_timeout = sum(_get_role_timeout(t.role) for t in tasks)
    if total_timeout > _SINGLE_AGENT_TIMEOUT_MS * 3:
        issues.append(PlanIssue(
            severity="warning",
            type="resource_waste",
            message=(
                f"Total estimated time {total_timeout // 60_000}min "
                f"exceeds 3x single agent baseline"
            ),
        ))

    # Verdict: any error => rejected
    has_errors = any(i.severity == "error" for i in issues)
    suggestion: str | None = None
    if has_errors:
        suggestion = "\n".join(
            f"- {i.message}" for i in issues if i.severity == "error"
        )

    return PlanValidation(
        approved=not has_errors,
        issues=issues,
        suggestion=suggestion,
    )
