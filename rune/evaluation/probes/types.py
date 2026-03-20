"""Probe base types for RUNE evaluation system.

Defines the abstract Probe interface that all evaluation probes implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rune.evaluation.types import ProbeResult


class Probe(ABC):
    """Abstract base class for evaluation probes.

    Each probe tests a specific capability of the agent system
    and returns a ProbeResult indicating success/failure and scoring.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this probe."""
        ...

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> ProbeResult:
        """Execute the probe and return the result.

        Parameters:
            context: Execution context (e.g., workspace root, tool registry).

        Returns:
            A ProbeResult with the probe outcome.
        """
        ...
