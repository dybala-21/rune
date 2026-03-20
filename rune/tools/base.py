"""Abstract Tool base class.

Ported from src/tools/base.ts - every tool must implement validate(),
simulate() (dry-run), execute(), and rollback().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rune.types import Domain, RiskLevel, ToolResult


class Tool(ABC):
    """Abstract base for all RUNE tools.

    Subclasses must define the five class-level properties and implement
    the four abstract methods.
    """

    # -- required properties (override in subclass) -------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """Domain this tool belongs to."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @property
    @abstractmethod
    def risk_level(self) -> RiskLevel:
        """Default risk level for operations."""

    @property
    @abstractmethod
    def actions(self) -> list[str]:
        """List of supported action names."""

    # -- abstract methods ---------------------------------------------------

    @abstractmethod
    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters before execution.

        Returns:
            Tuple of ``(valid, error_message)``.  When ``valid`` is True the
            error_message should be the empty string.
        """

    @abstractmethod
    async def simulate(self, params: dict[str, Any]) -> ToolResult:
        """Dry-run: preview the effect without making changes."""

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool action."""

    @abstractmethod
    async def rollback(self, rollback_data: dict[str, Any]) -> ToolResult:
        """Undo a previous execution given its rollback_data."""

    # -- default implementations --------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the tool is operational."""
        return True

    # -- helpers ------------------------------------------------------------

    def success(
        self,
        data: Any = None,
        rollback_data: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Build a successful ToolResult."""
        return ToolResult(
            success=True,
            data=data,
            rollback_data=rollback_data,
            duration_ms=0.0,
        )

    def failure(self, error: str) -> ToolResult:
        """Build a failed ToolResult."""
        return ToolResult(
            success=False,
            error=error,
            duration_ms=0.0,
        )


# ToolInfo - lightweight descriptor extracted from a Tool instance

@dataclass(slots=True)
class ToolInfo:
    """Lightweight snapshot of a Tool's metadata."""

    name: str
    domain: Domain
    description: str
    risk_level: RiskLevel
    actions: list[str] = field(default_factory=list)


def get_tool_info(tool: Tool) -> ToolInfo:
    """Extract a :class:`ToolInfo` from a live :class:`Tool` instance."""
    return ToolInfo(
        name=tool.name,
        domain=tool.domain,
        description=tool.description,
        risk_level=tool.risk_level,
        actions=list(tool.actions),
    )
