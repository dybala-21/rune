"""Skill signature generation and verification for RUNE.

Uses HMAC-SHA256 to sign and verify skill payloads, ensuring that skills
have not been tampered with since they were created or approved.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from rune.utils.fast_serde import json_encode

SIGNATURE_PREFIX = "hmac-sha256:"


def _canonicalize(
    name: str,
    description: str,
    body: str,
    scope: str,
    author: str,
) -> str:
    """Produce a deterministic JSON representation for signing."""
    return json_encode(
        {
            "name": name,
            "description": description,
            "body": body,
            "scope": scope,
            "author": author,
        }
    )


def _digest_hex(
    name: str,
    description: str,
    body: str,
    scope: str,
    author: str,
    secret: str,
) -> str:
    """Compute the HMAC-SHA256 hex digest of the canonical payload."""
    canonical = _canonicalize(name, description, body, scope, author)
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _normalize_signature(signature: str) -> str:
    trimmed = signature.strip()
    if trimmed.lower().startswith(SIGNATURE_PREFIX):
        return trimmed[len(SIGNATURE_PREFIX):]
    return trimmed


def sign_skill_payload(
    *,
    name: str,
    description: str,
    body: str,
    scope: str = "user",
    author: str = "",
    secret: str,
) -> str:
    """Generate an HMAC-SHA256 signature for a skill payload.

    Parameters:
        name: Skill name.
        description: Skill description.
        body: Skill body text.
        scope: ``"user"`` or ``"project"``.
        author: Author identifier.
        secret: HMAC secret key.

    Returns:
        Signature string prefixed with ``hmac-sha256:``.
    """
    digest = _digest_hex(name, description, body, scope, author, secret)
    return f"{SIGNATURE_PREFIX}{digest}"


def verify_skill_signature(
    *,
    name: str,
    description: str,
    body: str,
    scope: str = "user",
    author: str = "",
    secret: str,
    signature: str,
) -> bool:
    """Verify an HMAC-SHA256 signature for a skill payload.

    Uses constant-time comparison to prevent timing attacks.

    Parameters:
        name: Skill name.
        description: Skill description.
        body: Skill body text.
        scope: ``"user"`` or ``"project"``.
        author: Author identifier.
        secret: HMAC secret key.
        signature: The signature to verify.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    expected = _digest_hex(name, description, body, scope, author, secret)
    actual_hex = _normalize_signature(signature)

    # Validate hex format (64 hex characters for SHA-256)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", actual_hex):
        return False

    return hmac.compare_digest(expected.lower(), actual_hex.lower())
