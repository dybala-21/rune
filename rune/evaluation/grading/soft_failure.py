"""Soft vs hard failure classification for RUNE.

Classifies errors into soft failures (transient, retryable) and
hard failures (persistent, requiring user intervention).
"""

from __future__ import annotations

import re
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

FailureType = Literal["soft", "hard"]

# Patterns that indicate soft (transient / retryable) failures
_SOFT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"timed?\s*out", re.IGNORECASE),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too\s+many\s+requests", re.IGNORECASE),
    re.compile(r"429", re.IGNORECASE),
    re.compile(r"503", re.IGNORECASE),
    re.compile(r"502", re.IGNORECASE),
    re.compile(r"504", re.IGNORECASE),
    re.compile(r"connection\s+(reset|refused|timed)", re.IGNORECASE),
    re.compile(r"network\s+(error|unreachable)", re.IGNORECASE),
    re.compile(r"ECONNRESET", re.IGNORECASE),
    re.compile(r"ECONNREFUSED", re.IGNORECASE),
    re.compile(r"ETIMEDOUT", re.IGNORECASE),
    re.compile(r"temporary\s*(failure|error)", re.IGNORECASE),
    re.compile(r"server\s+overloaded", re.IGNORECASE),
    re.compile(r"retry\s+after", re.IGNORECASE),
    re.compile(r"service\s+unavailable", re.IGNORECASE),
]

# Patterns that indicate hard (persistent) failures
_HARD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"auth(entication|orization)\s*(failed|error|denied)", re.IGNORECASE),
    re.compile(r"401", re.IGNORECASE),
    re.compile(r"403", re.IGNORECASE),
    re.compile(r"invalid\s+(api\s+)?key", re.IGNORECASE),
    re.compile(r"billing", re.IGNORECASE),
    re.compile(r"quota\s+exceeded", re.IGNORECASE),
    re.compile(r"insufficient\s+(funds|credits|quota)", re.IGNORECASE),
    re.compile(r"invalid\s+(format|syntax|json|request)", re.IGNORECASE),
    re.compile(r"malformed", re.IGNORECASE),
    re.compile(r"schema\s+(validation|error)", re.IGNORECASE),
    re.compile(r"not\s+found", re.IGNORECASE),
    re.compile(r"404", re.IGNORECASE),
    re.compile(r"permission\s+denied", re.IGNORECASE),
    re.compile(r"type\s*error", re.IGNORECASE),
    re.compile(r"syntax\s*error", re.IGNORECASE),
    re.compile(r"name\s*error", re.IGNORECASE),
]


def classify_failure(error: str | Exception) -> FailureType:
    """Classify an error as soft (transient) or hard (persistent).

    Soft failures are typically retryable: timeouts, rate limits,
    transient network issues. Hard failures require user action:
    auth errors, billing issues, format errors, logic errors.

    Parameters:
        error: The error message or exception to classify.

    Returns:
        "soft" for transient/retryable errors, "hard" for persistent errors.
    """
    error_str = str(error)

    # Check hard patterns first (more definitive)
    for pattern in _HARD_PATTERNS:
        if pattern.search(error_str):
            log.debug("failure_classified", type="hard", pattern=pattern.pattern)
            return "hard"

    # Check soft patterns
    for pattern in _SOFT_PATTERNS:
        if pattern.search(error_str):
            log.debug("failure_classified", type="soft", pattern=pattern.pattern)
            return "soft"

    # Default: treat unknown errors as hard (conservative)
    log.debug("failure_classified", type="hard", reason="no_pattern_match")
    return "hard"
