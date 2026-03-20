"""Lesson extraction from task results.

Ported from src/memory/lessons.ts. Analyses completed tasks to extract
reusable lessons (failure patterns, success strategies, recurring patterns)
that can improve future agent performance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Data types

@dataclass(slots=True)
class LessonEntry:
    """A single extracted lesson from a task execution."""

    domain: str = ""
    lesson: str = ""
    confidence: float = 0.5
    source_goal: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# LLM Extractor protocol

@runtime_checkable
class LLMExtractor(Protocol):
    """Protocol for pluggable LLM-based lesson extraction.

    Implementations should call an LLM and return the generated text.
    """

    async def extract(self, prompt: str, max_tokens: int) -> str: ...


_LLM_EXTRACTION_PROMPT = """\
Analyze the following task execution result and extract reusable lessons.

Goal: {goal}
Intent: {intent}
Success: {success}
Error: {error}
Iterations: {iterations}

Return a JSON array of lesson objects. Each object must have:
- "domain": string - the domain/category (e.g. "coding", "debugging", "deployment")
- "lesson": string - the lesson learned (1-2 sentences)
- "confidence": number between 0 and 1
- "type": one of "success_pattern", "failure_pattern", "optimization", "warning"

Return ONLY the JSON array, no other text. Example:
[{{"domain": "debugging", "lesson": "Always check logs before restarting.", "confidence": 0.8, "type": "success_pattern"}}]
"""


async def extract_lessons_with_llm(
    goal: str,
    result: dict[str, Any],
    extractor: LLMExtractor,
    intent: str = "",
) -> list[LessonEntry]:
    """Use an LLM to extract lessons from a task result.

    Builds a structured prompt, calls the extractor, and parses the JSON
    response into LessonEntry objects. Falls back to an empty list on error.
    """
    prompt = _LLM_EXTRACTION_PROMPT.format(
        goal=_truncate(goal, 200),
        intent=intent or "unknown",
        success=result.get("success", False),
        error=_truncate(str(result.get("error", "")), 300),
        iterations=result.get("iterations", 1),
    )

    try:
        raw = await extractor.extract(prompt, max_tokens=1024)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        items = json_decode(raw)
        if not isinstance(items, list):
            items = [items]

        lessons: list[LessonEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            confidence = item.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            lessons.append(
                LessonEntry(
                    domain=str(item.get("domain", intent or "general")),
                    lesson=str(item.get("lesson", "")),
                    confidence=confidence,
                    source_goal=goal,
                )
            )

        log.debug(
            "llm_lessons_extracted",
            goal=goal[:80],
            count=len(lessons),
        )
        return lessons

    except Exception as exc:
        log.warning("llm_lesson_extraction_failed", error=str(exc))
        return []


# LessonExtractor

class LessonExtractor:
    """Extracts reusable lessons from task goals, results, and history."""

    def __init__(self) -> None:
        self._llm_extractor: LLMExtractor | None = None

    def set_llm_extractor(self, extractor: LLMExtractor) -> None:
        """Optionally enable LLM-based extraction."""
        self._llm_extractor = extractor

    def extract_from_result(
        self,
        goal: str,
        result: dict[str, Any],
        intent: str = "",
    ) -> list[LessonEntry]:
        """Analyse a task result and return any lessons learned.

        *result* is expected to have optional keys: ``success`` (bool),
        ``error`` (str), ``iterations`` (int), ``history`` (list).
        """
        lessons: list[LessonEntry] = []

        success = result.get("success", False)
        error = result.get("error", "")
        iterations = result.get("iterations", 1)
        history = result.get("history", [])

        if not success and error:
            failure_lesson = self._extract_failure_lesson(goal, error)
            if failure_lesson is not None:
                if intent:
                    failure_lesson.domain = intent
                lessons.append(failure_lesson)

        if success:
            success_lesson = self._extract_success_lesson(goal, iterations)
            if success_lesson is not None:
                if intent:
                    success_lesson.domain = intent
                lessons.append(success_lesson)

        if history and len(history) >= 2:
            pattern_lesson = self._extract_pattern_lesson(goal, history)
            if pattern_lesson is not None:
                if intent:
                    pattern_lesson.domain = intent
                lessons.append(pattern_lesson)

        if lessons:
            log.debug(
                "lessons_extracted",
                goal=goal[:80],
                count=len(lessons),
            )

        return lessons

    async def extract_from_result_async(
        self,
        goal: str,
        result: dict[str, Any],
        intent: str = "",
    ) -> list[LessonEntry]:
        """Async version of extract_from_result.

        If an LLM extractor is set, tries LLM-based extraction first.
        Falls back to heuristic extraction on any error.
        """
        if self._llm_extractor is not None:
            llm_lessons = await extract_lessons_with_llm(
                goal, result, self._llm_extractor, intent=intent,
            )
            if llm_lessons:
                return llm_lessons
            # Fallback to heuristic on empty/failed LLM result
            log.debug("llm_extraction_empty_fallback_to_heuristic", goal=goal[:80])

        return self.extract_from_result(goal, result, intent=intent)

    def _extract_failure_lesson(
        self, goal: str, error: str,
    ) -> LessonEntry | None:
        """Extract a lesson from a failed task."""
        if not error:
            return None

        # Identify common error categories
        error_lower = error.lower()

        if "permission" in error_lower or "access denied" in error_lower:
            lesson_text = (
                f"Task '{_truncate(goal)}' failed due to permissions. "
                "Verify access rights before attempting similar operations."
            )
            confidence = 0.8
        elif "timeout" in error_lower or "timed out" in error_lower:
            lesson_text = (
                f"Task '{_truncate(goal)}' timed out. "
                "Consider breaking into smaller steps or increasing timeout."
            )
            confidence = 0.7
        elif "not found" in error_lower or "no such file" in error_lower:
            lesson_text = (
                f"Task '{_truncate(goal)}' failed due to missing resource. "
                "Verify paths and prerequisites exist before proceeding."
            )
            confidence = 0.75
        elif "syntax" in error_lower or "parse" in error_lower:
            lesson_text = (
                f"Task '{_truncate(goal)}' encountered a syntax/parse error. "
                "Validate input format before execution."
            )
            confidence = 0.7
        else:
            # Generic failure lesson
            lesson_text = (
                f"Task '{_truncate(goal)}' failed with: {_truncate(error, 120)}. "
                "Review error details before retrying."
            )
            confidence = 0.5

        return LessonEntry(
            domain="failure",
            lesson=lesson_text,
            confidence=confidence,
            source_goal=goal,
        )

    def _extract_success_lesson(
        self, goal: str, iterations: int,
    ) -> LessonEntry | None:
        """Extract a lesson from a successful task based on iteration count."""
        if iterations <= 0:
            return None

        if iterations == 1:
            # First-try success - note the approach worked directly
            return LessonEntry(
                domain="efficiency",
                lesson=(
                    f"Task '{_truncate(goal)}' succeeded on first attempt. "
                    "The direct approach was effective."
                ),
                confidence=0.6,
                source_goal=goal,
            )

        if iterations <= 3:
            return LessonEntry(
                domain="efficiency",
                lesson=(
                    f"Task '{_truncate(goal)}' succeeded after {iterations} iterations. "
                    "Minor adjustments were needed."
                ),
                confidence=0.55,
                source_goal=goal,
            )

        # Many iterations - indicates difficulty
        return LessonEntry(
            domain="difficulty",
            lesson=(
                f"Task '{_truncate(goal)}' required {iterations} iterations. "
                "Consider breaking similar tasks into smaller steps."
            ),
            confidence=0.7,
            source_goal=goal,
        )

    def _extract_pattern_lesson(
        self, goal: str, history: list[Any],
    ) -> LessonEntry | None:
        """Detect recurring patterns in the task execution history.

        *history* is a list of step dicts, each optionally containing
        ``action`` (str) and ``result`` (str) keys.
        """
        if len(history) < 2:
            return None

        # Look for repeated actions (potential retry loops)
        actions = [
            str(step.get("action", "")) if isinstance(step, dict) else str(step)
            for step in history
        ]

        # Detect consecutive duplicates
        repeat_count = 0
        for i in range(1, len(actions)):
            if actions[i] and actions[i] == actions[i - 1]:
                repeat_count += 1

        if repeat_count >= 2:
            return LessonEntry(
                domain="pattern",
                lesson=(
                    f"Task '{_truncate(goal)}' showed repeated actions "
                    f"({repeat_count} repeats). This may indicate a retry loop — "
                    "consider alternative approaches when the same action fails twice."
                ),
                confidence=0.65,
                source_goal=goal,
            )

        # Detect error-then-recovery pattern
        error_recovery = 0
        for i in range(1, len(history)):
            prev = history[i - 1] if isinstance(history[i - 1], dict) else {}
            curr = history[i] if isinstance(history[i], dict) else {}
            prev_result = str(prev.get("result", "")).lower()
            curr_result = str(curr.get("result", "")).lower()
            if ("error" in prev_result or "fail" in prev_result) and (
                "success" in curr_result or "ok" in curr_result
            ):
                error_recovery += 1

        if error_recovery >= 1:
            return LessonEntry(
                domain="pattern",
                lesson=(
                    f"Task '{_truncate(goal)}' recovered from errors. "
                    "The recovery strategy was effective — note this approach "
                    "for similar error conditions."
                ),
                confidence=0.6,
                source_goal=goal,
            )

        return None


# Context building

def build_lessons_context(lessons: list[LessonEntry]) -> str:
    """Format lessons for agent prompt injection.

    Ported from LessonExtractor.buildLessonsContext in lessons.ts.
    Groups lessons by domain and caps at 5 to avoid prompt bloat.
    Returns empty string if no lessons are provided.
    """
    if not lessons:
        return ""

    # Cap at 5 lessons
    capped = lessons[:5]

    # Group by domain
    by_domain: dict[str, list[LessonEntry]] = {}
    for entry in capped:
        domain = entry.domain or "general"
        by_domain.setdefault(domain, []).append(entry)

    parts: list[str] = ["## Lessons from Similar Past Tasks\n"]

    for domain, entries in by_domain.items():
        if len(by_domain) > 1:
            parts.append(f"### {domain}")
        for entry in entries:
            conf_pct = f"{entry.confidence:.0%}"
            parts.append(f"- {entry.lesson} (confidence: {conf_pct})")
        parts.append("")

    return "\n".join(parts).rstrip("\n") + "\n"


def build_repeat_context(episode: dict[str, Any] | None) -> str:
    """Format a previous task episode for repeat execution.

    Ported from LessonExtractor.buildRepeatContext in lessons.ts.
    *episode* is a dict representation of a past task (typically from
    an Episode record or search result).  Expected keys:

    - ``domain``, ``action``, ``target`` - intent fields
    - ``task_summary`` - description of what was done
    - ``what_worked`` - list[str] of successful steps
    - ``lessons`` - list[str] of lessons learned
    - ``steps`` - list[dict] step summaries (optional)

    Returns empty string if episode is None or empty.
    """
    if not episode:
        return ""

    parts: list[str] = ["## Previous Task to Repeat\n"]

    if summary := episode.get("task_summary", ""):
        parts.append(f"Task: {summary}")

    if domain := episode.get("domain", ""):
        parts.append(f"Domain: {domain}")

    if action := episode.get("action", ""):
        parts.append(f"Action: {action}")

    if target := episode.get("target"):
        if isinstance(target, str):
            parts.append(f"Target: {target}")
        else:
            parts.append(f"Target: {json_encode(target)}")

    parts.append("")

    # What worked
    what_worked = episode.get("what_worked", [])
    if what_worked:
        parts.append("### Steps that worked:")
        for step in what_worked:
            parts.append(f"- {step}")
        parts.append("")

    # Step summary
    steps = episode.get("steps", [])
    if steps and not what_worked:
        parts.append("### Step summary:")
        for step in steps:
            if isinstance(step, dict):
                tool = step.get("tool", "")
                act = step.get("action", "")
                params = step.get("params", "")
                parts.append(f"- {tool}.{act}: {params}")
            else:
                parts.append(f"- {step}")
        parts.append("")

    # Lessons learned
    ep_lessons = episode.get("lessons", [])
    if ep_lessons:
        parts.append("### Lessons learned:")
        for lesson in ep_lessons:
            if isinstance(lesson, str):
                parts.append(f"- {lesson}")
            elif isinstance(lesson, dict):
                parts.append(f"- {lesson.get('lesson', lesson.get('content', str(lesson)))}")
            else:
                parts.append(f"- {lesson}")
        parts.append("")

    return "\n".join(parts).rstrip("\n") + "\n"


# Helpers

def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis if it exceeds *max_len*."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# Module singleton

_extractor: LessonExtractor | None = None


def get_lesson_extractor() -> LessonExtractor:
    global _extractor
    if _extractor is None:
        _extractor = LessonExtractor()
    return _extractor
