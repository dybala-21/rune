"""Only retrieved content can satisfy a source requirement."""

import pytest

from rune.agent.web_grounding import WebGrounding
from rune.types import CapabilityResult


@pytest.mark.parametrize("name,metadata,success,expected", [
    ("web_search", {"count": 1}, True, True),
    ("web_search", {"count": 0}, True, False),
    ("web_fetch", {"status_code": 200}, True, True),
    ("web_fetch", {"status_code": 403}, True, False),
    ("web_fetch", {"empty_content": True}, True, False),
    ("web_fetch", {}, False, False),
    ("browser_observe", {"url": "https://example.org"}, True, True),
    ("browser_navigate", {"url": "chrome-error://chromewebdata/"}, True, False),
    ("browser_navigate", {"url": "http://["}, True, False),
    ("browser_navigate", {"status": 404}, True, False),
    ("browser_extract", {"count": 0}, True, False),
    ("browser_profile", {}, True, False),
    ("browser_discover_apis", {}, True, False),
    ("browser_act", {}, True, False),
    ("think", {}, True, False),
    ("memory_search", {"count": 3}, True, False),
    ("file_read", {}, True, False),
    ("connector_request", {}, True, True),
])
def test_source_evidence(name, metadata, success, expected):
    check = WebGrounding(required=True)
    assert check.observe(name, {}, CapabilityResult(success=success, output="tool output", metadata=metadata)) is expected
    assert (check.blocker() is None) is expected


def test_empty_content_and_connector_writes_are_not_reads():
    check = WebGrounding(required=True)
    assert not check.observe("web_fetch", {}, CapabilityResult(success=True, output=" "))
    assert not check.observe("connector_request", {"method": "POST"}, CapabilityResult(success=True, output="sent"))
    assert check.blocker()


def test_optional_lookup_failure_cannot_silently_complete_as_chat():
    check = WebGrounding()
    assert check.blocker() is None
    check.observe("web_search", {}, CapabilityResult(success=False, error="unavailable"))
    assert check.blocker()
    check.observe("web_fetch", {}, CapabilityResult(success=True, output="retrieved facts"))
    assert check.blocker() is None


def test_optional_lookup_does_not_add_requirements_to_other_work():
    check = WebGrounding(require_on_lookup=False)
    check.observe("web_search", {}, CapabilityResult(success=False, error="unavailable"))
    assert check.blocker() is None


@pytest.mark.parametrize("metadata,success", [({"empty_content": True}, True), ({"status_code": 403}, False), ({"status_code": 200}, True)])
def test_filtered_search_needs_a_readable_source_page(metadata, success):
    check = WebGrounding(required=True)
    check.observe("web_search", {"freshness": "day"}, CapabilityResult(
        success=True, output="Indexed value: 267,250", metadata={"count": 1},
    ))
    assert check.blocker() and check.sources == 1 and check.pages == 0
    check.observe("web_fetch", {}, CapabilityResult(success=success, output="Page content", metadata=metadata))
    assert (check.blocker() is None) is (success and not metadata.get("empty_content", False))


def test_search_metadata_also_preserves_the_page_requirement():
    check = WebGrounding()
    check.observe("web_search", {}, CapabilityResult(success=True, output="Indexed value", metadata={"freshness": "day", "count": 1}))
    assert check.blocker()


def test_current_lookup_requires_a_page_without_search_freshness():
    check = WebGrounding(required=True, requires_page=True)
    check.observe("web_search", {}, CapabilityResult(success=True, output="Last month's status", metadata={"count": 1}))
    assert check.blocker()
    check.observe("web_fetch", {}, CapabilityResult(success=True, output="Current status and observation time"))
    assert check.blocker() is None


@pytest.mark.parametrize("answer,blocked", [
    ('Current quote: 269,500 KRW', True),
    ('현재가 **269,500원**', True),
    ('{"price":269500}', True),
    ('1. Quote unavailable. See https://example.org/269500', False),
    ('The requested 7 observations could not be retrieved.', False),
])
def test_opening_a_page_does_not_verify_numbers_from_search_excerpts(answer, blocked):
    check = WebGrounding(required=True, requires_page=True)
    check.observe("web_search", {}, CapabilityResult(success=True, output="1. Price 269,500; last 7 observations"))
    check.observe("web_fetch", {}, CapabilityResult(success=True, output="Quotes unavailable; 20 minute delay"))
    assert bool(check.blocker(answer, "Check the last 7 observations")) is blocked
    check.observe("web_fetch", {}, CapabilityResult(success=True, output='{"price":269500.0,"time":"2026-10-16T10:00:00+09:00"}'))
    assert check.blocker(answer, "Check the last 7 observations") is None


def test_source_numbers_preserve_precision_and_bound_exponents():
    from decimal import Decimal

    from rune.agent.source_values import numeric_literals

    assert numeric_literals('2.695e5 269500.00 269,500') == {Decimal('269500')}
    assert numeric_literals('12345678901234567890123456789') == {Decimal('12345678901234567890123456789')}
    assert numeric_literals('1e999999999') == set()


async def test_filtered_fetches_do_not_reuse_other_records_or_skip_validation():
    from rune.agent.cognitive_cache import SessionToolCache
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.capabilities.web import WebFetchParams

    filters = []

    async def fetch(params):
        filters.append(params.json_filter)
        return CapabilityResult(success=True, output=f"Record {params.json_filter}")

    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="web_fetch", description="Read page", parameters_model=WebFetchParams, execute=fetch))
    tool = build_tool_set(ToolAdapterOptions(enable_guardian=False, cognitive_cache=SessionToolCache()), registry=registry)["web_fetch"]
    first = await tool.function(url="https://example.org", jsonFilter="A")
    second = await tool.function(url="https://example.org", jsonFilter="B")
    repeated = await tool.function(url="https://example.org", jsonFilter="B")
    invalid = await tool.function(url="https://example.org", jsonFilter="B", maxLength="invalid")
    assert "Record A" in first and "Record B" in second and "Record B" in repeated
    assert "Invalid arguments" in invalid
    assert filters == ["A", "B"]


@pytest.mark.parametrize("args,expected", [({}, 8000), ({"maxLength": 20000}, 20000),
                                           ({"max_length": 3000}, 3000)])
async def test_small_fetch_default_can_expand_without_reusing_a_shorter_cache(args, expected):
    from rune.agent.cognitive_cache import SessionToolCache
    from rune.agent.tool_adapter import ToolAdapterOptions, build_tool_set
    from rune.capabilities.registry import CapabilityRegistry
    from rune.capabilities.types import CapabilityDefinition
    from rune.capabilities.web import WebFetchParams

    lengths = []

    async def fetch(params):
        lengths.append(params.max_length)
        return CapabilityResult(success=True, output="source excerpt", metadata={"truncated": True})

    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(name="web_fetch", description="Read page",
                                          parameters_model=WebFetchParams, execute=fetch))
    tool = build_tool_set(ToolAdapterOptions(enable_guardian=False, web_fetch_default_max_length=8000,
                                            cognitive_cache=SessionToolCache()), registry=registry)["web_fetch"]
    assert tool.json_schema["properties"]["maxLength"]["default"] == 8000
    output = await tool.function(url="https://example.org")
    assert "truncated" in str(output)
    await tool.function(url="https://example.org", **args)
    assert lengths == ([8000] if expected == 8000 else [8000, expected])
