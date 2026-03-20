"""Provider-native search tool resolver for RUNE.

Ported from src/capabilities/search/native-search.ts - resolves
LLM provider server-side web-search tools, with session/daily budget
tracking.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from rune.capabilities.search.provider_map import PROVIDER_SEARCH_MAP
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Budget tracker

class NativeSearchBudget:
    """Track session and daily native-search call counts."""

    def __init__(
        self,
        *,
        max_per_session: int | None = None,
        max_per_day: int | None = None,
    ) -> None:
        self._max_per_session = max_per_session
        self._max_per_day = max_per_day
        self._session_count = 0
        self._daily_count = 0
        self._daily_reset_date: str = date.today().isoformat()

    # -- helpers --

    def _reset_daily_if_needed(self) -> None:
        today = date.today().isoformat()
        if today != self._daily_reset_date:
            self._daily_count = 0
            self._daily_reset_date = today

    def can_search(self) -> bool:
        """Return whether budget allows another search."""
        self._reset_daily_if_needed()

        if self._max_per_session is not None and self._session_count >= self._max_per_session:
            log.debug(
                "native_search_budget_session",
                count=self._session_count,
                limit=self._max_per_session,
            )
            return False

        if self._max_per_day is not None and self._daily_count >= self._max_per_day:
            log.debug(
                "native_search_budget_daily",
                count=self._daily_count,
                limit=self._max_per_day,
            )
            return False

        return True

    def record(self) -> None:
        """Record that a native search was executed."""
        self._reset_daily_if_needed()
        self._session_count += 1
        self._daily_count += 1

    @property
    def counts(self) -> dict[str, int]:
        return {"session": self._session_count, "daily": self._daily_count}


# Resolver

async def resolve_native_search_tool(
    provider: str,
    *,
    native_search_enabled: bool = False,
    budget: NativeSearchBudget | None = None,
) -> dict[str, Any] | None:
    """Resolve a provider's native web-search tool.

    Returns a dict ``{"web_search_native": <tool_object>}`` when
    available, or ``None`` when native search is disabled / unsupported /
    over budget.

    The actual tool creation uses dynamic imports to handle missing
    SDK packages gracefully.
    """
    if not native_search_enabled:
        return None

    search_config = PROVIDER_SEARCH_MAP.get(provider)
    if not search_config or not search_config.get("has_native_search"):
        return None

    if budget and not budget.can_search():
        return None

    try:
        if provider == "openai":
            from openai import OpenAI  # type: ignore[import-untyped]  # noqa: F401
            if budget:
                budget.record()
            # Placeholder: actual integration depends on the AI SDK binding
            return {"web_search_native": {"provider": "openai", "type": "web_search"}}

        if provider == "anthropic":
            if budget:
                budget.record()
            return {"web_search_native": {"provider": "anthropic", "type": "web_search"}}

        log.debug("native_search_not_integrated", provider=provider)
        return None

    except ImportError as exc:
        log.debug("native_search_package_missing", provider=provider, error=str(exc))
        return None
