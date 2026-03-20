"""Search provider base types for RUNE.

Ported from src/capabilities/search/provider.ts - abstract search
provider interface and result dataclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    description: str | None = None
    age: str | None = None


@dataclass(slots=True)
class SearchOptions:
    """Options passed to every search provider."""
    query: str
    max_results: int = 10
    freshness: Literal["day", "week", "month", "year"] | None = None
    site: str | None = None


class SearchProvider(ABC):
    """Abstract base for all search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    async def search(self, options: SearchOptions) -> list[SearchResult]:
        """Execute a search and return results."""
