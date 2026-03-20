"""Tests for rune.api.cors_policy — ported from cors-policy.test.ts."""


from rune.api.cors_policy import resolve_cors_decision


class TestResolveCorsDecision:
    """Tests for resolve_cors_decision()."""

    def test_allows_same_origin_when_cors_not_configured(self):
        decision = resolve_cors_decision(
            request_origin="http://127.0.0.1:18789",
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is True
        assert decision.allow_origin is None

    def test_blocks_cross_origin_when_cors_not_configured(self):
        decision = resolve_cors_decision(
            request_origin="http://localhost:5173",
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is False

    def test_allows_explicit_origin_list_match_with_credentials(self):
        decision = resolve_cors_decision(
            configured_origin="http://localhost:5173, http://127.0.0.1:5173",
            request_origin="http://localhost:5173",
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is True
        assert decision.allow_origin == "http://localhost:5173"
        assert decision.allow_credentials is True

    def test_blocks_origin_not_in_configured_list(self):
        decision = resolve_cors_decision(
            configured_origin="http://localhost:5173",
            request_origin="https://evil.example.com",
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is False

    def test_supports_wildcard_origin_without_credentials(self):
        decision = resolve_cors_decision(
            configured_origin="*",
            request_origin="https://evil.example.com",
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is True
        assert decision.allow_origin == "*"
        assert decision.allow_credentials is False

    def test_allows_non_browser_requests_without_origin_header(self):
        decision = resolve_cors_decision(
            request_host="127.0.0.1:18789",
        )
        assert decision.allowed is True
        assert decision.allow_origin is None
