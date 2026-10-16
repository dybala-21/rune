"""Track source reads and numbers found only in search excerpts."""

from dataclasses import dataclass, field
from decimal import Decimal
from urllib.parse import urlsplit

from rune.agent.source_values import numeric_literals
from rune.types import CapabilityResult

_READS = {"web_search", "web_fetch", "browser_navigate", "browser_observe",
          "browser_extract", "browser_find", "browser_screenshot"}


@dataclass
class WebGrounding:
    required: bool = False
    require_on_lookup: bool = True
    attempts: int = 0
    sources: int = 0
    pages: int = 0
    requires_page: bool = False
    repairs: int = 0
    excerpt_values: set[Decimal] = field(default_factory=set)
    page_values: set[Decimal] = field(default_factory=set)

    def observe(self, name: str, params: dict, result: CapabilityResult) -> bool:
        if name not in _READS and not (name == "connector_request" and params.get("method", "GET").upper() == "GET"):
            return False
        if name == "web_fetch" and params.get("method", "GET").upper() != "GET":
            return False
        self.attempts += 1
        if self.require_on_lookup and name in {"web_search", "web_fetch"}:
            self.required = True
        metadata = result.metadata or {}
        # Cached reads do not add fresh evidence.
        if metadata.get("cached") or metadata.get("replayed"):
            return False
        if name == "web_search" and (params.get("freshness") or metadata.get("freshness")):
            self.requires_page = True
        if not result.success or not (result.output or "").strip():
            return False
        if (metadata.get("count") == 0 or metadata.get("empty_content") or metadata.get("observation_failed")
                or metadata.get("filter_no_match") or metadata.get("invalid_json") or metadata.get("delivery_truncated")):
            return False
        status = metadata.get("status_code", metadata.get("status"))
        if isinstance(status, int) and (status == 0 or status >= 400):
            return False
        url = metadata.get("url")
        if isinstance(url, str):
            try:
                if urlsplit(url).scheme not in {"http", "https"}:
                    return False
            except ValueError:
                return False
        self.sources += 1
        self.pages += int(name != "web_search")
        if self.requires_page:
            values = numeric_literals(result.output or "")
            (self.excerpt_values if name == "web_search" else self.page_values).update(values)
        return True

    def blocker(self, answer: str = "", request: str = "") -> str | None:
        if not self.required:
            return None
        if self.requires_page and not self.pages:
            return (
                "Only search excerpts are available for this time-sensitive lookup. Open a relevant page or API "
                "and check the observation date before reporting current values. A search freshness filter does "
                "not date the underlying facts. Do not repeat unverified prices or readings in the answer."
            )
        if self.requires_page and answer:
            unsupported = (numeric_literals(answer) & self.excerpt_values) - self.page_values - numeric_literals(request)
            if unsupported:
                return (
                    "These numbers occur in indexed search excerpts but not in any source page/API read in this run: "
                    + ", ".join(str(value) for value in sorted(unsupported)[:8]) + ". Opening a page without those observations does not verify them. "
                    "Try another relevant source within the tool budget. Do not present these numbers as current; "
                    "omit unresolved values and state the limitation instead of asking the user to supply an already available URL."
                )
        if self.sources:
            return None
        return (
            "External facts are needed, but no source has been retrieved in this run. "
            "Use the available search/fetch tools or read the relevant page. Memory and plans are not current evidence. "
            "If retrieval fails, explain the actual limitation and leave current facts unverified. "
            "Do not claim lookup is unavailable without trying it."
        )
