"""Quality gate -- validate proactive suggestion quality before delivery.

Checks for error patterns, low-confidence markers, and language-specific
quality signals in both English and Korean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Error patterns

_EN_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(error|fail(ed|ure)?|exception|crash|broken|invalid)\b", re.I),
    re.compile(r"\b(cannot|couldn't|unable|impossible)\b", re.I),
    re.compile(r"\b(undefined|null|NaN|NoneType)\b"),
    re.compile(r"\b(timeout|timed?\s*out|refused|denied)\b", re.I),
]

_KO_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(\uc2e4\ud328|\uc5d0\ub7ec|\uc624\ub958|\uc608\uc678|\ucda9\ub3cc)"),
    re.compile(r"(\ubd88\uac00\ub2a5|\ud560\s*\uc218\s*\uc5c6|\ubabb\s*\ud568|\uc548\s*\ub428)"),
    re.compile(r"(\uc2dc\uac04\s*\ucd08\uacfc|\uac70\ubd80|\uac70\uc808|\ub9cc\ub8cc)"),
    re.compile(r"(\uc798\ubabb\ub41c|\uc720\ud6a8\ud558\uc9c0\s*\uc54a|\uc190\uc0c1)"),
]

_LOW_CONFIDENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(maybe|perhaps|might|not sure|uncertain)\b", re.I),
    re.compile(r"(\uc544\ub9c8|\ud639\uc2dc|\ud655\uc2e4\ud558\uc9c0\s*\uc54a|\ubaa8\ub974\uaca0)"),
]


@dataclass(slots=True)
class QualityCheckResult:
    passed: bool
    reason: str = ""
    error_count: int = 0
    confidence_concern: bool = False


def check_quality(text: str, *, min_length: int = 10) -> QualityCheckResult:
    """Run quality checks on a proactive suggestion text.

    Returns a QualityCheckResult indicating whether the text passes
    quality gates for delivery.
    """
    if len(text.strip()) < min_length:
        return QualityCheckResult(passed=False, reason="Text too short")

    error_count = 0
    for pat in _EN_ERROR_PATTERNS + _KO_ERROR_PATTERNS:
        error_count += len(pat.findall(text))

    if error_count >= 3:
        return QualityCheckResult(
            passed=False,
            reason=f"Too many error indicators ({error_count})",
            error_count=error_count,
        )

    confidence_concern = any(pat.search(text) for pat in _LOW_CONFIDENCE_PATTERNS)

    return QualityCheckResult(
        passed=True,
        error_count=error_count,
        confidence_concern=confidence_concern,
    )
