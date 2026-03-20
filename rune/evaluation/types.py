"""Evaluation type definitions for RUNE.

Core types for the evaluation/probe system that tests agent capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(slots=True)
class ProbeResult:
    """Result of a single evaluation probe run."""

    probe_name: str = ""
    success: bool = False
    output: str = ""
    expected: str = ""
    duration_ms: float = 0.0
    score: float = 0.0


@dataclass(slots=True)
class EvaluationRun:
    """A complete evaluation run containing multiple probe results."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    probe_results: list[ProbeResult] = field(default_factory=list)
    total_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
