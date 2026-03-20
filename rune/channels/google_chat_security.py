"""Google Chat request verification.

Ported from src/channels/google-chat-security.ts - verifies bearer
tokens from Google Chat interaction requests using Google's public
certificate endpoints.  Supports both the recommended OIDC ID-token
audience mode and the project-number / Chat service-account JWT mode.
"""

from __future__ import annotations

import base64
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from rune.utils.fast_serde import json_decode

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.x509 import load_pem_x509_certificate

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTO = False

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False

# Constants

CHAT_ISSUER = "chat@system.gserviceaccount.com"
OIDC_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
OIDC_CERTS_URL = "https://www.googleapis.com/oauth2/v1/certs"
CHAT_ISSUER_CERTS_URL = (
    f"https://www.googleapis.com/service_accounts/v1/metadata/x509/{CHAT_ISSUER}"
)

# Types


@dataclass(slots=True, frozen=True)
class TokenVerificationResult:
    ok: bool
    payload: dict[str, Any] | None = None
    reason: str = ""


@dataclass(slots=True, frozen=True)
class TokenVerificationOptions:
    audience: str
    clock_skew_sec: int = 30


# Certificate cache

_cert_cache: dict[str, tuple[dict[str, str], float]] = {}


def _parse_max_age(cache_control: str | None) -> int | None:
    if not cache_control:
        return None
    for part in cache_control.split(","):
        part = part.strip().lower()
        if part.startswith("max-age="):
            try:
                return int(part[8:])
            except ValueError:
                return None
    return None


async def _fetch_certs(url: str) -> dict[str, str]:
    """Fetch (and cache) Google's public signing certificates."""
    now = time.time()
    cached = _cert_cache.get(url)
    if cached and cached[1] > now:
        return cached[0]

    if not _HAS_HTTPX:
        raise RuntimeError("httpx is required for Google Chat token verification")

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        certs: dict[str, str] = resp.json()
        max_age = _parse_max_age(resp.headers.get("cache-control")) or 300
        _cert_cache[url] = (certs, now + max_age)
        return certs


# JWT helpers

def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def _parse_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, str]:
    """Return (header, payload, signature_bytes, signed_portion)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    header = json_decode(_b64url_decode(parts[0]))
    payload = json_decode(_b64url_decode(parts[1]))
    signature = _b64url_decode(parts[2])
    signed = f"{parts[0]}.{parts[1]}"
    return header, payload, signature, signed


def _verify_audience(aud: Any, expected: str) -> bool:
    if isinstance(aud, str):
        return aud == expected
    if isinstance(aud, list):
        return expected in aud
    return False


def _certs_url_for_issuer(iss: str | None) -> str | None:
    if not iss:
        return None
    if iss == CHAT_ISSUER:
        return CHAT_ISSUER_CERTS_URL
    if iss in OIDC_ISSUERS:
        return OIDC_CERTS_URL
    return None


def _validate_time_claims(payload: dict[str, Any], clock_skew: int) -> str | None:
    now_sec = math.floor(time.time())
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and now_sec > exp + clock_skew:
        return "Token expired"
    nbf = payload.get("nbf")
    if isinstance(nbf, (int, float)) and now_sec + clock_skew < nbf:
        return "Token not active yet"
    return None


def _verify_rsa_signature(signed: str, signature: bytes, cert_pem: str) -> bool:
    """Verify an RSA-SHA256 signature against a PEM certificate."""
    if not _HAS_CRYPTO:
        raise RuntimeError(
            "cryptography package is required for Google Chat signature verification"
        )
    cert = load_pem_x509_certificate(cert_pem.encode("utf-8"))
    public_key = cert.public_key()
    try:
        public_key.verify(  # type: ignore[union-attr]
            signature,
            signed.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# Public API

async def verify_google_chat_bearer_token(
    token: str,
    *,
    audience: str,
    clock_skew_sec: int = 30,
) -> TokenVerificationResult:
    """Verify a bearer token from a Google Chat webhook request.

    Returns a :class:`TokenVerificationResult` indicating success or the
    reason for failure.
    """
    try:
        header, payload, signature, signed = _parse_jwt(token)
    except (ValueError, json.JSONDecodeError, Exception) as exc:
        return TokenVerificationResult(ok=False, reason=f"JWT parse error: {exc}")

    iss = payload.get("iss")
    certs_url = _certs_url_for_issuer(iss)
    if not certs_url:
        return TokenVerificationResult(ok=False, reason=f"Unrecognized issuer: {iss}")

    # Audience check
    if not _verify_audience(payload.get("aud"), audience):
        return TokenVerificationResult(ok=False, reason="Audience mismatch")

    # Time claims
    time_err = _validate_time_claims(payload, clock_skew_sec)
    if time_err:
        return TokenVerificationResult(ok=False, reason=time_err)

    # Issuer-specific: Chat service account must have email claim
    if iss == CHAT_ISSUER:
        email = payload.get("email")
        if email != CHAT_ISSUER:
            email_verified = payload.get("email_verified", False)
            if not email_verified:
                return TokenVerificationResult(
                    ok=False, reason="Chat issuer email not verified"
                )

    # Fetch certs and verify signature
    kid = header.get("kid")
    try:
        certs = await _fetch_certs(certs_url)
    except Exception as exc:
        return TokenVerificationResult(ok=False, reason=f"Failed to fetch certs: {exc}")

    cert_pem = certs.get(kid or "") if kid else None
    if not cert_pem:
        # Try all certs if kid is missing
        verified = False
        for pem in certs.values():
            if _verify_rsa_signature(signed, signature, pem):
                verified = True
                break
        if not verified:
            return TokenVerificationResult(ok=False, reason="Signature verification failed")
    else:
        if not _verify_rsa_signature(signed, signature, cert_pem):
            return TokenVerificationResult(ok=False, reason="Signature verification failed")

    return TokenVerificationResult(ok=True, payload=payload)
