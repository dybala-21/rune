"""Cost estimation tests, including prompt caching discount."""

from __future__ import annotations

import pytest

from rune.ui.cost import estimate_cost, format_cost


class TestEstimateCostBasic:
    """Basic cost estimation without caching."""

    def test_known_model(self):
        # claude-opus-4-6: $0.015/1K input, $0.075/1K output
        cost = estimate_cost("claude-opus-4-6", 1000, 1000)
        assert cost == pytest.approx(0.015 + 0.075)

    def test_unknown_model(self):
        cost = estimate_cost("unknown-model", 1000, 1000)
        assert cost == 0.0

    def test_zero_tokens(self):
        cost = estimate_cost("claude-opus-4-6", 0, 0)
        assert cost == 0.0

    def test_local_model_free(self):
        cost = estimate_cost("llama3:70b", 100000, 100000)
        assert cost == 0.0


class TestEstimateCostWithCaching:
    """Cost estimation with Anthropic prompt caching."""

    def test_cache_read_discount(self):
        # 10K tokens, all from cache = 0.1x input rate
        # claude-opus-4-6: $0.015/1K input
        # Without cache: 10K * $0.015/1K = $0.15
        # With cache read: 10K * $0.015/1K * 0.1 = $0.015
        cost_no_cache = estimate_cost("claude-opus-4-6", 10000, 0)
        cost_cached = estimate_cost(
            "claude-opus-4-6", 10000, 0, cached_input_tokens=10000
        )
        assert cost_no_cache == pytest.approx(0.15)
        assert cost_cached == pytest.approx(0.015)
        assert cost_cached == pytest.approx(cost_no_cache * 0.1)

    def test_cache_write_surcharge(self):
        # 10K tokens written to cache = 1.25x input rate
        # Without cache: $0.15
        # With cache write: 10K * $0.015/1K * 1.25 = $0.1875
        cost_no_cache = estimate_cost("claude-opus-4-6", 10000, 0)
        cost_write = estimate_cost(
            "claude-opus-4-6", 10000, 0, cache_write_tokens=10000
        )
        assert cost_no_cache == pytest.approx(0.15)
        assert cost_write == pytest.approx(0.1875)
        assert cost_write == pytest.approx(cost_no_cache * 1.25)

    def test_mixed_cache_and_uncached(self):
        # 15K input total: 5K uncached + 8K cache read + 2K cache write
        # Input rate: $0.015/1K
        # Cost = 5K*0.015 + 8K*0.015*0.1 + 2K*0.015*1.25
        #      = 0.075 + 0.012 + 0.0375 = 0.1245
        cost = estimate_cost(
            "claude-opus-4-6",
            15000,
            0,
            cached_input_tokens=8000,
            cache_write_tokens=2000,
        )
        expected = (5000 / 1000 * 0.015) + (8000 / 1000 * 0.015 * 0.1) + (2000 / 1000 * 0.015 * 1.25)
        assert cost == pytest.approx(expected)

    def test_cache_with_output(self):
        # 10K cached input + 1K output
        # claude-opus-4-6: $0.015/1K input, $0.075/1K output
        # Cost = 10K * 0.015 * 0.1 + 1K * 0.075 = 0.015 + 0.075 = 0.09
        cost = estimate_cost(
            "claude-opus-4-6", 10000, 1000, cached_input_tokens=10000
        )
        assert cost == pytest.approx(0.015 + 0.075)

    def test_backward_compatible_without_cache_args(self):
        # Calling without cache args should work identically to old behavior
        cost1 = estimate_cost("claude-opus-4-6", 5000, 2000)
        cost2 = estimate_cost(
            "claude-opus-4-6", 5000, 2000, cached_input_tokens=0, cache_write_tokens=0
        )
        assert cost1 == cost2


class TestFormatCost:
    def test_small_cost(self):
        assert format_cost(0.0001) == "$0.0001"
        assert format_cost(0.0099) == "$0.0099"

    def test_medium_cost(self):
        assert format_cost(0.0234) == "$0.0234"
        assert format_cost(0.9999) == "$0.9999"

    def test_large_cost(self):
        assert format_cost(1.50) == "$1.50"
        assert format_cost(12.34) == "$12.34"
