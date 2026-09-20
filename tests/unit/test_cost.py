"""Published rates, incomplete usage, and per-request billing boundaries."""

from types import SimpleNamespace

import pytest

from rune.llm.pricing import estimate_request_cost, usage_payload
from rune.llm.usage import token_counts
from rune.ui.cost import estimate_cost, format_cost


def counts(input_tokens=1000, output_tokens=1000, **extra):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_write_reported": True, **extra}


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5", .030), ("anthropic/claude-opus-4-6", .030),
    ("xai/grok-4.6", .008), ("vertex_ai/gemini-2.5-flash", .0028),
    ("openai/responses/gpt-6-astra", .060), ("gpt-4o", .0125),
    ("ollama/llama3:70b", 0),
])
def test_published_standard_rates(model, expected):
    assert estimate_cost(model, 1000, 1000) == pytest.approx(expected)


def test_unknown_usage_never_looks_free():
    assert estimate_cost("unknown-model", 1000, 1000) is None
    assert estimate_request_cost("gpt-6-astra", {"input_tokens": 1000, "output_tokens": 20}) is None
    assert format_cost(None) == "Unavailable"
    assert format_cost(0) == "$0.0000"


def test_cache_reads_and_both_anthropic_write_durations():
    usage = token_counts({"input_tokens": 5000, "output_tokens": 1000,
                         "cache_read_input_tokens": 8000, "cache_creation_input_tokens": 2000,
                         "cache_creation": {"ephemeral_1h_input_tokens": 500}})
    assert usage["input_tokens"] == 15000
    assert estimate_request_cost("claude-opus-5", usage) == pytest.approx(
        (5000 * 5 + 8000 * .5 + 1500 * 6.25 + 500 * 10 + 1000 * 25) / 1e6)
    assert estimate_cost("claude-opus-4-6", 10000, 0, cached_input_tokens=10000) == pytest.approx(.005)
    assert estimate_cost("claude-opus-4-6", 10000, 0, cache_write_tokens=10000) == pytest.approx(.0625)


@pytest.mark.parametrize("model,threshold,normal,large", [
    ("xai/grok-4.6", 199999, (2, 6), (4, 12)),
    ("gpt-6-astra", 272000, (10, 50), (20, 75)),
    ("gemini/gemini-2.5-pro", 200000, (1.25, 10), (2.5, 15)),
])
def test_long_context_boundary(model, threshold, normal, large):
    assert estimate_request_cost(model, counts(threshold, 100)) == pytest.approx(
        (threshold * normal[0] + 100 * normal[1]) / 1e6)
    assert estimate_request_cost(model, counts(threshold + 1, 100)) == pytest.approx(
        ((threshold + 1) * large[0] + 100 * large[1]) / 1e6)


@pytest.mark.parametrize("options", [{"api_base": "https://custom.invalid"},
    {"extra_body": {"service_tier": "priority"}}, {"inference_geo": "us"},
    {"messages": [{"content": [{"type": "input_audio", "input_audio": {}}]}]}])
def test_unsupported_billing_options_stay_unpriced(options):
    assert estimate_request_cost("claude-opus-5", counts(), options) is None


def test_fast_mode_and_invalid_counts():
    assert estimate_request_cost("claude-opus-5", counts(), {"speed": "fast"}) == pytest.approx(.06)
    assert estimate_request_cost("claude-opus-5", counts(cached_input_tokens=1001)) is None
    assert estimate_request_cost("claude-opus-5", counts(cache_write_tokens=-1)) is None


def test_old_saved_usage_is_not_repriced_or_shown_as_zero():
    usage = {"calls": 1, "reported_calls": 1, "input_tokens": 100, "output_tokens": 20,
             "total_tokens": 120, "cached_input_tokens": 0, "cache_write_tokens": 0}
    payload = usage_payload(SimpleNamespace(timings={"usage": usage}))
    assert payload["cost"]["usd"] is None
    assert payload["cost"]["unpricedCalls"] == 1


def test_regional_vertex_prices_are_not_silently_treated_as_global():
    model = "vertex_ai/gemini-3.1-pro-preview"
    assert estimate_request_cost(model, counts(), {"vertex_location": "us-central1"}) is None
    assert estimate_request_cost(model, counts(), {"vertex_location": "global"}) == pytest.approx(.014)


def test_anthropic_write_without_a_read_field_is_included_in_input():
    usage = token_counts({"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 300})
    assert usage["input_tokens"] == 400 and usage["cache_write_tokens"] == 300
