"""Search providers for RUNE.

Ported from src/capabilities/search/ - pluggable search backends
with rate limiting and native provider support.
"""

from rune.capabilities.search.brave import BraveSearchProvider
from rune.capabilities.search.browser_search import BrowserSearchProvider
from rune.capabilities.search.duckduckgo import DuckDuckGoSearchProvider
from rune.capabilities.search.native_search import NativeSearchBudget, resolve_native_search_tool
from rune.capabilities.search.provider import SearchOptions, SearchProvider, SearchResult
from rune.capabilities.search.provider_map import PROVIDER_SEARCH_MAP, get_native_search_providers
from rune.capabilities.search.rate_limiter import SearchRateLimiter

__all__ = [
    "BraveSearchProvider",
    "BrowserSearchProvider",
    "DuckDuckGoSearchProvider",
    "NativeSearchBudget",
    "PROVIDER_SEARCH_MAP",
    "SearchOptions",
    "SearchProvider",
    "SearchRateLimiter",
    "SearchResult",
    "get_native_search_providers",
    "resolve_native_search_tool",
]
