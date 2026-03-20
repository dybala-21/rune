"""Intelligent timing & intent analysis for RUNE proactive suggestions.

Computes a composite timing score from urgency, relevance, receptivity,
and value dimensions. Provides event-specific thresholds and intent
signal detection from user command/file patterns.

Ported from src/proactive/intelligence.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.proactive.context import AwarenessContext
from rune.proactive.types import EngagementMetrics, Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Score weights (aligned with TS SCORE_WEIGHTS)

_W_URGENCY = 0.25
_W_RELEVANCE = 0.3
_W_RECEPTIVITY = 0.25
_W_VALUE = 0.2

# Event-type base urgency (TS EVENT_URGENCY)
EVENT_URGENCY: dict[str, float] = {
    "task_failed": 0.9,
    "task_completed": 0.3,
    "idle_detected": 0.2,
    "file_change": 0.4,
    "context_switch": 0.3,
    "time_trigger": 0.5,
}

# Per-event shouldIntervene thresholds (TS EVENT_THRESHOLD)
EVENT_THRESHOLD: dict[str, float] = {
    "task_failed": 0.35,
    "task_completed": 0.45,
    "time_trigger": 0.40,
    "context_switch": 0.45,
    "file_change": 0.50,
    "idle_detected": 0.50,
}

_DEFAULT_THRESHOLD = 0.55


# Data structures


@dataclass(slots=True)
class OpportuneScore:
    """4-factor score with reasoning trail."""

    urgency: float = 0.0
    relevance: float = 0.0
    receptivity: float = 0.0
    value: float = 0.0
    total: float = 0.0
    reasoning: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IntentSignal:
    """A detected intent from user activity patterns."""

    intent: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggestion: str | None = None


@dataclass(slots=True)
class WorkflowContext:
    """Inferred workflow phase and predicted next actions."""

    current_phase: str = "idle"  # starting | working | reviewing | finishing | idle
    likely_next_actions: list[str] = field(default_factory=list)
    focus_area: str | None = None


# Composite scoring


def compute_opportune_score(
    suggestion: Suggestion,
    context: AwarenessContext,
    engagement: EngagementMetrics,
    *,
    event_type: str = "",
    recent_commands: list[str] | None = None,
) -> OpportuneScore:
    """Compute a 4-factor opportune score with reasoning.

    Parameters:
        suggestion: The candidate suggestion.
        context: Current environment/awareness context.
        engagement: User engagement metrics.
        event_type: The event that triggered evaluation (e.g. "task_failed").
        recent_commands: Recent user commands for intent detection.

    Returns:
        An OpportuneScore with per-dimension breakdown and reasoning.
    """
    reasoning: list[str] = []

    urgency = _calculate_urgency(suggestion, context, event_type, reasoning)
    relevance = _calculate_relevance(suggestion, context, event_type, reasoning)
    receptivity = _calculate_receptivity(engagement, context, reasoning)
    value = _calculate_value(suggestion, context, event_type, reasoning)

    total = (
        urgency * _W_URGENCY
        + relevance * _W_RELEVANCE
        + receptivity * _W_RECEPTIVITY
        + value * _W_VALUE
    )

    score = OpportuneScore(
        urgency=round(urgency, 4),
        relevance=round(relevance, 4),
        receptivity=round(receptivity, 4),
        value=round(value, 4),
        total=round(total, 4),
        reasoning=reasoning,
    )

    log.debug(
        "opportune_score",
        type=suggestion.type,
        event_type=event_type,
        total=score.total,
    )
    return score


def should_intervene(
    score: OpportuneScore,
    event_type: str = "",
    *,
    signals: list[IntentSignal] | None = None,
    min_score_override: float | None = None,
) -> bool:
    """Decide whether to intervene based on score and event-specific thresholds.

    Strong intent signals (confidence >= 0.7) lower the threshold by 0.1.
    """
    threshold = min_score_override or EVENT_THRESHOLD.get(event_type, _DEFAULT_THRESHOLD)

    # Strong signals lower the bar
    if signals:
        max_conf = max((s.confidence for s in signals), default=0.0)
        if max_conf >= 0.7:
            threshold = max(0.2, threshold - 0.1)

    return score.total >= threshold


# Per-dimension scoring


def _calculate_urgency(
    suggestion: Suggestion,
    context: AwarenessContext,
    event_type: str,
    reasoning: list[str],
) -> float:
    """Event-specific urgency (0-1)."""
    # Event-based base
    if event_type and event_type in EVENT_URGENCY:
        base = EVENT_URGENCY[event_type]
        reasoning.append(f"urgency: event={event_type} base={base:.2f}")
    else:
        # Fall back to suggestion type
        type_scores: dict[str, float] = {
            "warning": 0.9, "reminder": 0.7, "followup": 0.5,
            "optimization": 0.3, "insight": 0.2,
        }
        base = type_scores.get(suggestion.type, 0.3)
        reasoning.append(f"urgency: type={suggestion.type} base={base:.2f}")

    # Idle duration adjustment (from context time_context)
    if event_type == "idle_detected":
        idle_min = context.time_context.get("idle_minutes", 0)
        if idle_min > 30:
            base = min(1.0, base + 0.4)
            reasoning.append(f"urgency: idle {idle_min}min → +0.4")
        elif idle_min > 10:
            base = min(1.0, base + 0.2)
            reasoning.append(f"urgency: idle {idle_min}min → +0.2")

    # Task completion: boost if many tasks completed recently
    if event_type == "task_completed":
        task_count = context.time_context.get("recent_task_count", 0)
        if task_count >= 3:
            base = min(1.0, base + 0.3)
            reasoning.append(f"urgency: {task_count} recent tasks → +0.3")

    # Time trigger: morning or end-of-day boost
    if event_type == "time_trigger":
        hour = context.time_context.get("hour", 12)
        if 7 <= hour <= 9:
            base = max(base, 0.6)
            reasoning.append("urgency: morning time trigger")
        elif 16 <= hour <= 18:
            base = max(base, 0.7)
            reasoning.append("urgency: end-of-day time trigger")

    # Expiry boost
    if suggestion.expires_at is not None:
        from datetime import datetime

        remaining = (suggestion.expires_at - datetime.now()).total_seconds()
        if remaining < 300:
            base = min(1.0, base + 0.2)
            reasoning.append("urgency: expiring <5min")
        elif remaining < 3600:
            base = min(1.0, base + 0.1)
            reasoning.append("urgency: expiring <1hr")

    return base


def _calculate_relevance(
    suggestion: Suggestion,
    context: AwarenessContext,
    event_type: str,
    reasoning: list[str],
) -> float:
    """Context-aligned relevance (0-1)."""
    score = 0.3
    reasoning.append("relevance: baseline=0.30")

    # File match
    if suggestion.source and context.recent_files:
        for f in context.recent_files[:5]:
            if suggestion.source in f or f in suggestion.description:
                score += 0.3
                reasoning.append("relevance: source matches recent file +0.3")
                break

    # Git context
    if context.git_status and suggestion.type in ("reminder", "warning"):
        score += 0.2
        reasoning.append("relevance: git active + reminder/warning +0.2")

    # Activity mode
    if context.user_activity_mode == "debug" and suggestion.type == "warning":
        score += 0.2
        reasoning.append("relevance: debug mode + warning +0.2")
    elif context.user_activity_mode == "acceleration" and suggestion.type == "optimization":
        score += 0.15
        reasoning.append("relevance: acceleration + optimization +0.15")

    # Task-related events have inherent relevance
    if event_type == "task_failed":
        score += 0.3
        reasoning.append("relevance: task_failed inherently relevant +0.3")
    elif event_type == "context_switch":
        score += 0.15
        reasoning.append("relevance: context_switch +0.15")

    # High confidence suggestion
    if suggestion.confidence > 0.7:
        score += 0.1
        reasoning.append("relevance: high confidence +0.1")

    return min(1.0, score)


def _calculate_receptivity(
    engagement: EngagementMetrics,
    context: AwarenessContext,
    reasoning: list[str],
) -> float:
    """User receptivity (0-1) from engagement and time-of-day."""
    if engagement.suggestions_shown == 0:
        reasoning.append("receptivity: no history, default=0.5")
        base = 0.5
    else:
        rate = engagement.acceptance_rate
        base = 0.3 + rate * 0.4
        reasoning.append(f"receptivity: acceptance_rate={rate:.2f} → {base:.2f}")

    # Time-of-day adjustment
    hour = context.time_context.get("hour", 12)
    if hour >= 22 or hour < 7:
        base *= 0.7
        reasoning.append("receptivity: late night penalty *0.7")
    elif 12 <= hour <= 13:
        base *= 0.85
        reasoning.append("receptivity: lunch break penalty *0.85")

    # Deep work detection: if user has been coding intensively, reduce
    if context.user_activity_mode == "deep_work":
        base *= 0.6
        reasoning.append("receptivity: deep work mode penalty *0.6")

    return max(0.1, min(1.0, base))


def _calculate_value(
    suggestion: Suggestion,
    context: AwarenessContext,
    event_type: str,
    reasoning: list[str],
) -> float:
    """Intrinsic suggestion value (0-1)."""
    base = min(1.0, max(0.0, suggestion.confidence))
    reasoning.append(f"value: suggestion confidence={base:.2f}")

    # Event-specific value adjustments
    if event_type == "task_failed":
        base = max(base, 0.9)
        reasoning.append("value: task_failed → min 0.9")
    elif event_type == "task_completed":
        base = max(base, 0.5)
        reasoning.append("value: task_completed → min 0.5")
    elif event_type == "file_change":
        base = max(base, 0.7)
        reasoning.append("value: file_change → min 0.7")

    return base


# Intent signal detection


def detect_intent_signals(
    context: AwarenessContext,
    recent_commands: list[str] | None = None,
) -> list[IntentSignal]:
    """Detect user intent from recent command and file patterns.

    Detects:
    - preparing_commit: 2+ git commands
    - searching_for_something: 2+ search/grep/find commands
    - struggling_with_task: 2+ failed commands (error patterns)
    """
    signals: list[IntentSignal] = []
    commands = recent_commands or []

    if len(commands) < 2:
        return signals

    recent = [c.lower() for c in commands[-8:]]

    # Preparing commit
    git_cmds = [c for c in recent if c.startswith("git ")]
    if len(git_cmds) >= 2:
        signals.append(IntentSignal(
            intent="preparing_commit",
            confidence=min(0.9, 0.5 + len(git_cmds) * 0.1),
            evidence=git_cmds[:3],
            suggestion="Would you like help reviewing changes before committing?",
        ))

    # Searching for something
    search_cmds = [c for c in recent if any(
        kw in c for kw in ("grep", "find", "rg ", "ag ", "search")
    )]
    if len(search_cmds) >= 2:
        signals.append(IntentSignal(
            intent="searching_for_something",
            confidence=min(0.8, 0.4 + len(search_cmds) * 0.15),
            evidence=search_cmds[:3],
            suggestion="I can help search the codebase more efficiently.",
        ))

    # Struggling with task (repeated failures)
    fail_patterns = ("error", "failed", "not found", "permission denied", "exit code")
    fail_cmds = [c for c in recent if any(p in c for p in fail_patterns)]
    if len(fail_cmds) >= 2:
        signals.append(IntentSignal(
            intent="struggling_with_task",
            confidence=min(0.85, 0.5 + len(fail_cmds) * 0.12),
            evidence=fail_cmds[:3],
            suggestion="It looks like you're hitting errors. Can I help debug?",
        ))

    return signals


def infer_workflow_context(context: AwarenessContext) -> WorkflowContext:
    """Infer current workflow phase from time and environment signals.

    Phases:
    - starting: early morning, few recent files
    - working: active file modifications
    - reviewing: git status shows changes
    - finishing: late in day, git activity
    - idle: no recent activity
    """
    hour = context.time_context.get("hour", 12)
    has_git = bool(context.git_status)
    file_count = len(context.recent_files)

    if file_count == 0 and not has_git:
        return WorkflowContext(current_phase="idle", likely_next_actions=["start_task"])

    if 6 <= hour <= 9 and file_count < 3:
        return WorkflowContext(
            current_phase="starting",
            likely_next_actions=["review_tasks", "check_messages", "start_coding"],
        )

    if has_git and (16 <= hour <= 19):
        return WorkflowContext(
            current_phase="finishing",
            likely_next_actions=["commit", "push", "create_pr", "update_docs"],
            focus_area="wrap_up",
        )

    if has_git:
        return WorkflowContext(
            current_phase="reviewing",
            likely_next_actions=["run_tests", "commit", "diff_review"],
            focus_area="verification",
        )

    return WorkflowContext(
        current_phase="working",
        likely_next_actions=["edit_code", "run_tests", "search_docs"],
        focus_area="implementation",
    )


# Backward-compatible alias

# Keep the old TimingScore name available
TimingScore = OpportuneScore


def compute_timing_score(
    suggestion: Suggestion,
    context: AwarenessContext,
    engagement: EngagementMetrics,
) -> OpportuneScore:
    """Backward-compatible wrapper for compute_opportune_score."""
    return compute_opportune_score(suggestion, context, engagement)
