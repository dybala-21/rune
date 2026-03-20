"""Tests for rune.channels.google_chat_security — JWT parsing, audience check, token verification."""

from __future__ import annotations

import base64
import json
import math
import time
from unittest.mock import AsyncMock, patch

import pytest

from rune.channels.google_chat_security import (
    OIDC_ISSUERS,
    _b64url_decode,
    _parse_jwt,
    _validate_time_claims,
    _verify_audience,
    verify_google_chat_bearer_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_unsigned_jwt(payload: dict) -> str:
    """Build a JWT without a real signature (for parse / audience tests)."""
    header = {"alg": "RS256", "typ": "JWT", "kid": "test-kid"}
    h = _b64url_encode(json.dumps(header).encode())
    p = _b64url_encode(json.dumps(payload).encode())
    sig = _b64url_encode(b"\x00" * 32)
    return f"{h}.{p}.{sig}"


# ---------------------------------------------------------------------------
# _b64url_decode
# ---------------------------------------------------------------------------


class TestB64UrlDecode:
    def test_decode_padded(self):
        original = b"hello world"
        encoded = base64.urlsafe_b64encode(original).rstrip(b"=").decode()
        assert _b64url_decode(encoded) == original


# ---------------------------------------------------------------------------
# _parse_jwt
# ---------------------------------------------------------------------------


class TestParseJwt:
    def test_parses_three_part_token(self):
        token = _make_unsigned_jwt({"iss": "test", "aud": "example"})
        header, payload, sig_bytes, signed = _parse_jwt(token)
        assert header["alg"] == "RS256"
        assert payload["iss"] == "test"
        assert payload["aud"] == "example"
        assert "." in signed

    def test_raises_on_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid JWT"):
            _parse_jwt("not.a.valid.jwt.token")

    def test_raises_on_two_parts(self):
        with pytest.raises(ValueError):
            _parse_jwt("header.payload")


# ---------------------------------------------------------------------------
# _verify_audience
# ---------------------------------------------------------------------------


class TestVerifyAudience:
    def test_string_audience_match(self):
        assert _verify_audience("https://example.com", "https://example.com") is True

    def test_string_audience_mismatch(self):
        assert _verify_audience("https://other.com", "https://example.com") is False

    def test_list_audience_match(self):
        assert _verify_audience(["a", "b", "c"], "b") is True

    def test_list_audience_mismatch(self):
        assert _verify_audience(["a", "b"], "c") is False

    def test_none_audience(self):
        assert _verify_audience(None, "any") is False


# ---------------------------------------------------------------------------
# _validate_time_claims
# ---------------------------------------------------------------------------


class TestValidateTimeClaims:
    def test_valid_token(self):
        now = math.floor(time.time())
        assert _validate_time_claims({"exp": now + 300, "iat": now - 10}, 30) is None

    def test_expired_token(self):
        now = math.floor(time.time())
        result = _validate_time_claims({"exp": now - 100}, 30)
        assert result is not None
        assert "expired" in result.lower()

    def test_not_yet_active_token(self):
        now = math.floor(time.time())
        result = _validate_time_claims({"nbf": now + 1000}, 30)
        assert result is not None
        assert "not active" in result.lower()


# ---------------------------------------------------------------------------
# verify_google_chat_bearer_token — integration-level (mocked certs)
# ---------------------------------------------------------------------------


class TestVerifyGoogleChatBearerToken:
    @pytest.mark.asyncio
    async def test_rejects_unrecognized_issuer(self):
        token = _make_unsigned_jwt({"iss": "bad-issuer", "aud": "test"})
        result = await verify_google_chat_bearer_token(token, audience="test")
        assert result.ok is False
        assert "issuer" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rejects_audience_mismatch(self):
        now = math.floor(time.time())
        issuer = next(iter(OIDC_ISSUERS))
        token = _make_unsigned_jwt({
            "iss": issuer,
            "aud": "https://wrong.com",
            "exp": now + 300,
        })
        # Patch _fetch_certs so it doesn't make real HTTP calls
        with patch("rune.channels.google_chat_security._fetch_certs", new_callable=AsyncMock, return_value={}):
            result = await verify_google_chat_bearer_token(
                token, audience="https://correct.com",
            )
        assert result.ok is False
        assert "audience" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rejects_malformed_token(self):
        result = await verify_google_chat_bearer_token("not-a-jwt", audience="test")
        assert result.ok is False
        assert "parse" in result.reason.lower() or "JWT" in result.reason
