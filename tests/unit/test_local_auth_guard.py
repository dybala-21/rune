"""Tests for rune.api.local_auth_guard — ported from local-auth-guard.test.ts."""


from rune.api.local_auth_guard import is_trusted_local_bypass_request


class TestIsTrustedLocalBypassRequest:
    """Tests for is_trusted_local_bypass_request()."""

    def test_allows_non_browser_local_requests_without_origin_referer(self):
        assert is_trusted_local_bypass_request({}, 18789) is True

    def test_allows_same_port_loopback_origin(self):
        headers = {"origin": "http://127.0.0.1:18789"}
        assert is_trusted_local_bypass_request(headers, 18789) is True

    def test_blocks_different_port_loopback_origin(self):
        headers = {"origin": "http://127.0.0.1:9999"}
        assert is_trusted_local_bypass_request(headers, 18789) is False

    def test_blocks_non_loopback_origin(self):
        headers = {"origin": "https://evil.example.com"}
        assert is_trusted_local_bypass_request(headers, 18789) is False

    def test_blocks_cross_site_fetch_metadata(self):
        headers = {
            "origin": "http://127.0.0.1:18789",
            "sec-fetch-site": "cross-site",
        }
        assert is_trusted_local_bypass_request(headers, 18789) is False

    def test_blocks_non_loopback_referer_when_origin_absent(self):
        headers = {"referer": "https://evil.example.com/page"}
        assert is_trusted_local_bypass_request(headers, 18789) is False

    def test_allows_localhost_origin(self):
        headers = {"origin": "http://localhost:18789"}
        assert is_trusted_local_bypass_request(headers, 18789) is True

    def test_allows_same_origin_sec_fetch_site(self):
        headers = {
            "origin": "http://127.0.0.1:18789",
            "sec-fetch-site": "same-origin",
        }
        assert is_trusted_local_bypass_request(headers, 18789) is True
