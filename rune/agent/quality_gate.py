"""Quality gate - sub-agent result quality verification.

Ported from src/agent/quality-gate.ts (116 lines) - heuristic quality
checks to prevent hollow success (success=true but no real content).

Checks:
1. Hollow answer — content too short
2. Executor evidence - executor role with no action traces
3. Error masking - success=true with failure keywords
4. Suspicious speed - unrealistically fast completion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types


@dataclass(slots=True)
class QualityCheck:
    """Result of a quality gate check."""

    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    suggestion: str | None = None


@dataclass(slots=True)
class AgentResult:
    """Minimal agent result for quality checking."""

    success: bool
    answer: str = ""
    iterations: int = 0
    duration_ms: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TaskInfo:
    """Minimal task info for quality checking."""

    id: str = ""
    role: str = "executor"
    goal: str = ""


# Error masking patterns

ERROR_MASKING_PATTERNS: list[str] = [
    "could not",
    "couldn't",
    "unable to",
    "failed to",
    "cannot",
    "can't",
    "error occurred",
    "exception",
    "not possible",
    "impossible",
    # Korean error patterns
    "실패",        # failure
    "에러",        # error
    "오류",        # error/fault
    "불가능",      # impossible
    "할 수 없",    # cannot
    "시간 초과",   # timeout
    "거부",        # denied/rejected
]


# check_task_quality

def check_task_quality(
    task: TaskInfo,
    result: AgentResult,
    *,
    threshold: float = 0.3,
) -> QualityCheck:
    """Heuristic quality verification for sub-agent results.

    Failed results are not subject to quality gate - only success=true
    results are checked for hollow success patterns.
    """
    # Failed results pass through
    if not result.success:
        return QualityCheck(passed=True, score=1.0)

    issues: list[str] = []
    worst_score = 1.0
    answer = result.answer or ""

    # 1. Hollow answer detection
    if len(answer) < 50:
        issues.append(
            f"Response too short ({len(answer)} chars) — "
            f"should include concrete results"
        )
        worst_score = min(worst_score, 0.2)

    # 2. Executor action evidence
    if task.role == "executor":
        has_action_evidence = (
            result.iterations >= 3
            or any(h.get("type") == "action" for h in result.history)
        )
        if not has_action_evidence:
            issues.append(
                "Executor role but insufficient action evidence "
                "(iterations < 3, no action entries)"
            )
            worst_score = min(worst_score, 0.2)

    # 3. Error masking detection (success=true with failure keywords)
    lower_answer = answer.lower()
    masked_errors = [p for p in ERROR_MASKING_PATTERNS if p in lower_answer]
    if len(masked_errors) >= 2:
        issues.append(
            f"Reported success but error keywords found: "
            f"{', '.join(masked_errors[:3])}"
        )
        worst_score = min(worst_score, 0.5)

    # 4. Suspicious speed (executor/researcher completing in < 2 seconds)
    if result.duration_ms < 2000 and task.role in ("executor", "researcher"):
        issues.append(
            f"{task.role} completed in {result.duration_ms}ms — "
            f"may not have performed actual work"
        )
        worst_score = min(worst_score, 0.6)

    # 5. Research quality (file output with few sources)
    if task.role == "researcher" and result.success and len(answer) > 500:
        # Check for residual draft markers
        draft_markers = ["[TODO]", "[TBD]", "검증 필요", "추가 검증", "리서치 반영 예정"]
        found_markers = [m for m in draft_markers if m in answer]
        if found_markers:
            issues.append(
                f"Draft markers still present: {', '.join(found_markers)}"
            )
            worst_score = min(worst_score, 0.5)

        # Check source count (URLs or citation patterns)
        import re
        url_count = len(re.findall(r"https?://\S+", answer))
        citation_count = len(re.findall(r"\[\d+\]|\(\d{4}\)", answer))
        source_count = max(url_count, citation_count)
        if source_count < 3 and len(answer) > 2000:
            issues.append(
                f"Long research output ({len(answer)} chars) with only "
                f"{source_count} sources"
            )
            worst_score = min(worst_score, 0.6)

    # Verdict
    passed = worst_score >= threshold
    suggestion: str | None = None
    if not passed:
        lines = [
            "Quality issues detected in previous attempt:",
            *[f"- {i}" for i in issues],
            "",
            "Please retry with concrete, substantive results.",
        ]
        if task.role == "executor":
            lines.append("You must execute tools and include execution results.")
        else:
            lines.append("Include specific data and sources.")
        suggestion = "\n".join(lines)

    return QualityCheck(
        passed=passed,
        score=worst_score,
        issues=issues,
        suggestion=suggestion,
    )
