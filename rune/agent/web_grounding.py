"""Track source reads and numbers found only in search excerpts."""

from dataclasses import dataclass, field
from decimal import Decimal
from urllib.parse import urlsplit

from rune.agent.source_context import SourceContext
from rune.agent.source_records import SourceRecords, SourceReview
from rune.agent.source_values import numeric_literals
from rune.types import CapabilityResult

_READS = {"web_search", "web_fetch", "browser_navigate", "browser_observe",
          "browser_extract", "browser_find", "browser_screenshot"}


def _original_fields(receipt) -> bool:
    return receipt is None or (isinstance(receipt, dict) and isinstance(receipt.get("query"), dict)
                               and receipt["query"].get("operation") == "latest")


@dataclass
class WebGrounding:
    required: bool = False
    require_on_lookup: bool = True
    attempts: int = 0
    sources: int = 0
    pages: int = 0
    requires_page: bool = False
    repairs: int = 0
    check_records: bool = False
    records: SourceRecords = field(default_factory=SourceRecords)
    excerpt_values: set[Decimal] = field(default_factory=set)
    page_values: set[Decimal] = field(default_factory=set)
    context: SourceContext = field(default_factory=SourceContext)
    reused_sources: int = 0
    review: SourceReview = field(default_factory=SourceReview)

    def restore(self, context: SourceContext) -> None:
        if self.requires_page:
            return
        self.context = context
        for entry in context.entries:
            self.reused_sources += 1
            if "json" in entry["content_type"].lower() and _original_fields(entry.get("record_query")):
                self.records.observe(entry["body"], entry["id"])
            else:
                self.records.complete = False

    def snapshot(self, answer: str, request: str = "") -> dict:
        review = self.records.review(answer) if self.check_records else SourceReview()
        return {**review.snapshot(), "freshSources": self.sources, "reusedSources": self.reused_sources,
                "excerptOnlyNumbers": self.excerpt_only_numbers(answer, request)}

    def excerpt_only_numbers(self, answer: str, request: str = "") -> list[str]:
        if not self.requires_page:
            return []
        values = (numeric_literals(answer) & self.excerpt_values) - self.page_values - numeric_literals(request)
        return [str(value) for value in sorted(values)[:8]]

    def observe(self, name: str, params: dict, result: CapabilityResult) -> bool:
        metadata = result.metadata or {}
        captured_body = name == "browser_discover_apis" and metadata.get("captured_body") is True
        if name not in _READS and not captured_body and not (name == "connector_request" and params.get("method", "GET").upper() == "GET"):
            return False
        if name == "web_fetch" and params.get("method", "GET").upper() != "GET":
            return False
        self.attempts += 1
        if self.require_on_lookup and name in {"web_search", "web_fetch"}:
            self.required = True
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
        body = result.output or ""
        if name != "web_search":
            offset = metadata.get("body_offset", 0)
            if type(offset) is not int or not 0 <= offset <= len(body):
                return False
            body = body[offset:]
            if not body.strip():
                return False
        self.sources += 1
        self.pages += int(name != "web_search")
        if name != "web_search":
            import hashlib

            source_url = metadata.get("url") or params.get("url") or ""
            source_id = hashlib.sha256((str(source_url) + "\n" + body).encode()).hexdigest()
            # Aggregates are computed results, not fields copied from the response.
            if ("json" in metadata.get("content_type", "").lower()
                    and not metadata.get("truncated") and _original_fields(metadata.get("record_query"))):
                self.records.observe(body, source_id)
            else:
                self.records.complete = False
            # Browser and connector snapshots may depend on permissions that later expire.
            if name == "web_fetch" and not metadata.get("truncated"):
                self.context.capture(source_url, body, metadata.get("content_type", "text/plain"),
                                     query=metadata.get("record_query"))
        if self.requires_page:
            values = numeric_literals(body)
            (self.excerpt_values if name == "web_search" else self.page_values).update(values)
        return True

    def blocker(self, answer: str = "", request: str = "") -> str | None:
        if not self.required:
            return None
        if self.check_records and answer:
            self.review = self.records.review(answer)
            if self.review.status == "contradicted":
                return self.review.detail
        if self.requires_page and not self.pages:
            return (
                "Only search excerpts are available for this time-sensitive lookup. Open a relevant page or API "
                "and check the observation date before reporting current values. A search freshness filter does "
                "not date the underlying facts. Do not repeat unverified prices or readings in the answer."
            )
        # A numeric mention alone cannot distinguish an assertion from a caveat.
        if self.sources or self.reused_sources:
            return None
        return (
            "External facts are needed, but no source has been retrieved in this run. "
            "Use the available search/fetch tools or read the relevant page. Memory and plans are not current evidence. "
            "If retrieval fails, explain the actual limitation and leave current facts unverified. "
            "Do not claim lookup is unavailable without trying it."
        )
