"""Reflexion / introspection learning for RUNE.

Records task outcomes per domain, tracks suggestion rejections,
and adaptively adjusts proactive strategy thresholds.

Ported from src/proactive/reflexion.ts.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

_learner: ReflexionLearner | None = None

_MAX_LESSONS_PER_DOMAIN = 50
_MAX_REJECTION_HISTORY = 50
_CONSECUTIVE_WINDOW_SEC = 1800  # 30 minutes
_CONSECUTIVE_THRESHOLD = 5  # Conservative: require 5 rejections before adjusting


# Data structures


@dataclass(slots=True)
class RejectionRecord:
    """Record of a rejected suggestion."""

    timestamp: float  # monotonic
    event_type: str
    suggestion_type: str
    score: float
    reason: str = "unknown"  # bad_timing | irrelevant | too_frequent | not_helpful | user_busy
    context_summary: str = ""


@dataclass(slots=True)
class StrategyAdjustment:
    """Adaptive threshold adjustments learned from rejections."""

    min_score_threshold: float | None = None
    min_interval_seconds: float | None = None
    disabled_event_types: list[str] = field(default_factory=list)
    reasoning: str = ""


# Reflexion Learner


class ReflexionLearner:
    """Learns from task outcomes and suggestion rejections.

    Tracks:
    - Task outcomes per domain with lesson extraction
    - Suggestion rejection history with reason classification
    - Adaptive strategy: score threshold, min interval, disabled events
    """

    __slots__ = (
        "_outcomes",
        "_lessons",
        "_rejections",
        "_strategy",
        "_domain_stats",
    )

    def __init__(self) -> None:
        self._outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lessons: dict[str, list[str]] = defaultdict(list)
        self._rejections: list[RejectionRecord] = []
        self._strategy = StrategyAdjustment()
        # domain → {"success": int, "failure": int}
        self._domain_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )

    # -- Strategy getters ---------------------------------------------------

    def get_score_threshold(self) -> float | None:
        """Return the current min score threshold override, or None for default."""
        return self._strategy.min_score_threshold

    def get_min_interval(self) -> float:
        """Return the current min interval between interventions (seconds)."""
        return self._strategy.min_interval_seconds or 120.0

    def is_event_disabled(self, event_type: str) -> bool:
        """Check if an event type has been disabled by reflexion learning."""
        return event_type in self._strategy.disabled_event_types

    # -- Task outcome tracking ----------------------------------------------

    def record_task_outcome(self, outcome: dict[str, Any]) -> None:
        """Record a task outcome for a given domain.

        Expected outcome keys: domain, success, goal, error, steps_taken, duration_ms
        """
        domain = outcome.get("domain", "general")
        self._outcomes[domain].append(outcome)

        # Update domain stats
        success = outcome.get("success", False)
        self._domain_stats[domain]["success" if success else "failure"] += 1

        # Extract lesson
        lesson = self._extract_lesson(outcome)
        if lesson:
            lessons = self._lessons[domain]
            if lesson not in lessons:
                lessons.append(lesson)
                if len(lessons) > _MAX_LESSONS_PER_DOMAIN:
                    self._lessons[domain] = lessons[-_MAX_LESSONS_PER_DOMAIN:]
                log.info("reflexion_lesson", domain=domain, lesson=lesson)

    def get_domain_lessons(self, domain: str) -> list[str]:
        """Return all accumulated lessons for a domain."""
        return list(self._lessons.get(domain, []))

    def get_domain_success_rate(self, domain: str) -> float:
        """Return success rate for a domain (0-1), or 1.0 if no data."""
        stats = self._domain_stats.get(domain)
        if not stats:
            return 1.0
        total = stats["success"] + stats["failure"]
        if total == 0:
            return 1.0
        return stats["success"] / total

    def _extract_lesson(self, outcome: dict[str, Any]) -> str | None:
        """Extract a reusable lesson from a task outcome."""
        domain = outcome.get("domain", "general")
        success = outcome.get("success", False)
        goal = outcome.get("goal", "")
        error = outcome.get("error")
        steps = outcome.get("steps_taken", 0)

        if not goal:
            return None

        if not success and error:
            # Categorize failure
            err_lower = error.lower()
            if "timeout" in err_lower:
                return f"[{domain}] Timeout on '{goal}' — consider increasing limits."
            if "permission" in err_lower:
                return f"[{domain}] Permission denied on '{goal}' — check access rights."
            if "not found" in err_lower:
                return f"[{domain}] Resource not found for '{goal}' — verify paths/names."
            return f"[{domain}] Failed: '{goal}' — Error: {error}"

        if success and steps > 10:
            return (
                f"[{domain}] Completed '{goal}' but took {steps} steps; "
                f"consider a more direct approach."
            )

        if success and steps <= 2:
            return f"[{domain}] Efficiently completed '{goal}' in {steps} step(s)."

        return None

    # -- Rejection tracking -------------------------------------------------

    def record_rejection(
        self,
        event_type: str,
        suggestion_type: str,
        score: float,
        *,
        reason: str = "",
        context_summary: str = "",
    ) -> None:
        """Record a suggestion rejection and check for consecutive pattern."""
        if not reason:
            reason = self._classify_rejection(score)

        record = RejectionRecord(
            timestamp=time.monotonic(),
            event_type=event_type,
            suggestion_type=suggestion_type,
            score=score,
            reason=reason,
            context_summary=context_summary,
        )
        self._rejections.append(record)

        # Trim history
        if len(self._rejections) > _MAX_REJECTION_HISTORY:
            self._rejections = self._rejections[-_MAX_REJECTION_HISTORY:]

        log.info(
            "reflexion_rejection",
            event=event_type,
            reason=reason,
            score=round(score, 3),
        )

        # Auto-adjust on consecutive rejections
        self._check_consecutive_rejections()

    def _classify_rejection(self, score: float) -> str:
        """Rule-based rejection reason from score breakdown."""
        # Simple heuristic - could be enhanced with OpportuneScore components
        now = time.monotonic()
        recent = [r for r in self._rejections if now - r.timestamp < 120]
        if len(recent) >= 2:
            return "too_frequent"
        if score < 0.3:
            return "bad_timing"
        if score < 0.4:
            return "irrelevant"
        return "unknown"

    def _check_consecutive_rejections(self) -> None:
        """Auto-adjust strategy if 3+ rejections in 30 min.

        Increases min_interval_seconds * 1.5 (max 900s)
        Increases min_score_threshold + 0.05 (max 0.8)
        """
        now = time.monotonic()
        recent = [r for r in self._rejections if now - r.timestamp < _CONSECUTIVE_WINDOW_SEC]

        if len(recent) < _CONSECUTIVE_THRESHOLD:
            return

        # Increase interval
        current_interval = self._strategy.min_interval_seconds or 120.0
        new_interval = min(900.0, current_interval * 1.5)
        self._strategy.min_interval_seconds = new_interval

        # Increase threshold
        current_threshold = self._strategy.min_score_threshold or 0.55
        new_threshold = min(0.8, current_threshold + 0.05)
        self._strategy.min_score_threshold = new_threshold

        # Log event types with high rejection counts (but don't auto-disable
        # yet - threshold adjustment is sufficient for Phase −1b).
        type_counts: dict[str, int] = defaultdict(int)
        for r in recent:
            type_counts[r.event_type] += 1
        for evt, count in type_counts.items():
            if count >= _CONSECUTIVE_THRESHOLD:
                log.warning(
                    "reflexion_high_rejection_event",
                    event_type=evt,
                    rejections=count,
                    note="not auto-disabled; threshold adjustment only",
                )

        self._strategy.reasoning = (
            f"Auto-adjusted after {len(recent)} rejections in {_CONSECUTIVE_WINDOW_SEC}s: "
            f"interval={new_interval:.0f}s, threshold={new_threshold:.2f}"
        )

        log.info(
            "reflexion_strategy_adjusted",
            interval=new_interval,
            threshold=new_threshold,
            disabled=self._strategy.disabled_event_types,
        )

    # -- LLM-based analysis -------------------------------------------------

    async def analyze_with_llm(
        self,
        rejection_text: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze a rejection using an LLM for deeper insight.

        Falls back to rule-based analysis if the LLM call fails.
        """
        ctx = context or {}
        domain = ctx.get("domain", "general")
        event_type = ctx.get("event_type", "unknown")
        score_total = ctx.get("score", 0.0)

        prompt = (
            "You are an AI behavior analyst. A proactive suggestion was rejected.\n"
            "Analyze the rejection and provide a JSON object with:\n"
            '{"reason": "<bad_timing|irrelevant|too_frequent|not_helpful|user_busy|unknown>",\n'
            ' "lesson": "<one-sentence lesson>",\n'
            ' "confidence": <0.0-1.0>}\n\n'
            f"Event type: {event_type}\n"
            f"Domain: {domain}\n"
            f"Score: {score_total}\n"
            f"Rejected suggestion: {rejection_text}\n"
        )

        try:
            from rune.llm.client import get_llm_client

            client = get_llm_client()
            response = await client.generate(prompt)
            import re

            from rune.utils.fast_serde import json_decode

            json_match = re.search(r"\{[^{}]*\}", response)
            if json_match:
                parsed = json_decode(json_match.group())
                return {
                    "reason": parsed.get("reason", "unknown"),
                    "lesson": parsed.get("lesson", ""),
                    "confidence": float(parsed.get("confidence", 0.5)),
                }
        except Exception as exc:
            log.debug("reflexion_llm_fallback", error=str(exc))

        return self._rule_based_analysis(rejection_text, ctx)

    def _rule_based_analysis(
        self,
        rejection_text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Rule-based fallback for rejection analysis."""
        text_lower = rejection_text.lower()
        reason = "unknown"
        lesson = ""

        if any(w in text_lower for w in ("busy", "later", "not now")):
            reason = "bad_timing"
            lesson = "User was busy; defer to a quieter moment."
        elif any(w in text_lower for w in ("irrelevant", "not related", "wrong")):
            reason = "irrelevant"
            lesson = "Suggestion did not match user context."
        elif any(w in text_lower for w in ("again", "stop", "too many")):
            reason = "too_frequent"
            lesson = "Reduce intervention frequency."
        elif any(w in text_lower for w in ("useless", "not helpful", "obvious")):
            reason = "not_helpful"
            lesson = "Increase suggestion quality threshold."

        return {"reason": reason, "lesson": lesson, "confidence": 0.4}

    # -- Statistics ---------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated rejection and outcome statistics."""
        now = time.monotonic()
        recent = [r for r in self._rejections if now - r.timestamp < 3600]

        # Top reasons
        reason_counts: dict[str, int] = defaultdict(int)
        event_counts: dict[str, int] = defaultdict(int)
        for r in self._rejections:
            reason_counts[r.reason] += 1
            event_counts[r.event_type] += 1

        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_rejections": len(self._rejections),
            "recent_rejections": len(recent),
            "top_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
            "top_event_types": [{"event": e, "count": c} for e, c in top_events],
            "strategy": {
                "min_score_threshold": self._strategy.min_score_threshold,
                "min_interval_seconds": self._strategy.min_interval_seconds,
                "disabled_event_types": self._strategy.disabled_event_types,
                "reasoning": self._strategy.reasoning,
            },
            "domain_stats": dict(self._domain_stats),
        }

    def reset(self) -> None:
        """Clear all history and reset strategy to defaults."""
        self._outcomes.clear()
        self._lessons.clear()
        self._rejections.clear()
        self._domain_stats.clear()
        self._strategy = StrategyAdjustment()
        log.info("reflexion_reset")


def get_reflexion_learner() -> ReflexionLearner:
    """Get or create the singleton ReflexionLearner."""
    global _learner
    if _learner is None:
        _learner = ReflexionLearner()
    return _learner
