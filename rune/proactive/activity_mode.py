"""CHI 2025 3-mode activity detection for RUNE.

Classifies user activity into acceleration, exploration, or debug modes
based on recent tool usage patterns, error counts, and step progression.
"""

from __future__ import annotations

from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

ActivityMode = Literal["acceleration", "exploration", "debug"]

# Patterns that indicate focused, sequential work (edit -> test -> edit)
_ACCELERATION_TOOLS = {"file.write", "file.edit", "bash.execute", "project.build"}
_ACCELERATION_SEQUENCES = [
    ("file.edit", "bash.execute", "file.edit"),
    ("file.write", "bash.execute", "file.write"),
    ("file.edit", "project.build", "file.edit"),
]

# Patterns that indicate broad exploration
_EXPLORATION_TOOLS = {"file.read", "web.search", "project.search", "web.fetch"}

# Minimum error count for debug mode classification
_DEBUG_ERROR_THRESHOLD = 2

# Repeated-tool threshold for debug mode
_DEBUG_REPEAT_THRESHOLD = 3

_detector: ActivityModeDetector | None = None


class ActivityModeDetector:
    """Detects the user's current activity mode from tool-call patterns.

    Modes:
        acceleration: Sequential focused work (edit -> test -> edit pattern).
        exploration: Broad reading (many file.read, web.search calls).
        debug: Error-heavy sessions (bash failures, repeated command patterns).
    """

    __slots__ = ()

    def detect(
        self,
        recent_tools: list[str],
        error_count: int,
        step: int,
    ) -> ActivityMode:
        """Classify activity mode from recent context.

        Parameters:
            recent_tools: Ordered list of recent tool names.
            error_count: Number of errors in the recent window.
            step: Current step number in the agent loop.

        Returns:
            The detected activity mode.
        """
        if not recent_tools:
            return "exploration"

        # --- Debug mode: high error count or repeated failing commands ---
        if error_count >= _DEBUG_ERROR_THRESHOLD:
            log.debug("activity_mode_debug", reason="high_error_count", errors=error_count)
            return "debug"

        # Check for repeated tool invocations (sign of frustration / debug loop)
        if len(recent_tools) >= _DEBUG_REPEAT_THRESHOLD:
            last_n = recent_tools[-_DEBUG_REPEAT_THRESHOLD:]
            if len(set(last_n)) == 1 and last_n[0] in ("bash.execute", "file.edit"):
                log.debug("activity_mode_debug", reason="repeated_pattern", tool=last_n[0])
                return "debug"

        # --- Acceleration mode: sequential focused sequences ---
        if len(recent_tools) >= 3:
            recent_triple = tuple(recent_tools[-3:])
            if recent_triple in _ACCELERATION_SEQUENCES:
                log.debug("activity_mode_acceleration", reason="sequence_match", seq=recent_triple)
                return "acceleration"

        # Count focused-work tools vs exploration tools
        acceleration_count = sum(1 for t in recent_tools if t in _ACCELERATION_TOOLS)
        exploration_count = sum(1 for t in recent_tools if t in _EXPLORATION_TOOLS)

        total = len(recent_tools)
        accel_ratio = acceleration_count / total
        explore_ratio = exploration_count / total

        if accel_ratio >= 0.6:
            log.debug("activity_mode_acceleration", reason="ratio", ratio=accel_ratio)
            return "acceleration"

        if explore_ratio >= 0.5:
            log.debug("activity_mode_exploration", reason="ratio", ratio=explore_ratio)
            return "exploration"

        # Default: early steps tend to be exploratory
        if step <= 3:
            return "exploration"

        return "acceleration"


def get_activity_mode_detector() -> ActivityModeDetector:
    """Get or create the singleton ActivityModeDetector."""
    global _detector
    if _detector is None:
        _detector = ActivityModeDetector()
    return _detector
