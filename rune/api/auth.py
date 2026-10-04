"""Generate, persist and validate API bearer tokens."""

import contextlib
import json
import os
import secrets
import time
from typing import Any

from starlette.requests import Request

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_TOKEN_FILE = rune_home() / "api-tokens.json"
_TOKEN_PREFIX = "rune_"
_TOKEN_BYTE_LENGTH = 32


# Token storage

def _load_tokens() -> dict[str, dict[str, Any]]:
    """Load tokens from disk."""
    if not _TOKEN_FILE.exists():
        return {}
    try:
        data = json_decode(_TOKEN_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("token_load_failed", error=str(exc))
        return {}


def _save_tokens(tokens: dict[str, dict[str, Any]]) -> None:
    """Persist tokens to disk."""
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(
        json.dumps(tokens, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Restrict file permissions (owner read/write only)
    with contextlib.suppress(OSError):
        _TOKEN_FILE.chmod(0o600)


# Public API

def generate_token(*, label: str = "", expires_seconds: int | None = None) -> str:
    """Create and store a token; expires_seconds=None leaves it valid indefinitely."""
    raw = secrets.token_urlsafe(_TOKEN_BYTE_LENGTH)
    token = f"{_TOKEN_PREFIX}{raw}"

    tokens = _load_tokens()
    entry: dict[str, Any] = {
        "created_at": time.time(),
        "label": label,
    }
    if expires_seconds is not None:
        entry["expires_at"] = time.time() + expires_seconds
    tokens[token] = entry
    _save_tokens(tokens)

    log.info("token_generated", label=label)
    return token


def verify_token(token: str) -> bool:
    """Verify that a token is valid and not expired."""
    if not token or not token.startswith(_TOKEN_PREFIX):
        return False

    tokens = _load_tokens()
    entry = tokens.get(token)
    if entry is None:
        return False

    # Check expiration
    expires_at = entry.get("expires_at")
    if expires_at is not None and time.time() > expires_at:
        # Token expired - remove it
        tokens.pop(token, None)
        _save_tokens(tokens)
        return False

    return True


def revoke_token(token: str) -> bool:
    """Revoke (delete) a token. Returns True if the token existed."""
    tokens = _load_tokens()
    if token in tokens:
        del tokens[token]
        _save_tokens(tokens)
        log.info("token_revoked")
        return True
    return False


def list_tokens() -> list[dict[str, Any]]:
    """List all tokens (without exposing full token values)."""
    tokens = _load_tokens()
    result: list[dict[str, Any]] = []
    for tok, entry in tokens.items():
        result.append({
            "token_prefix": tok[:12] + "...",
            "label": entry.get("label", ""),
            "created_at": entry.get("created_at"),
            "expires_at": entry.get("expires_at"),
        })
    return result


# FastAPI dependency

def token_required() -> bool:
    from rune.cloud.boundary import hosted

    return hosted() or os.environ.get("RUNE_REQUIRE_TOKEN") == "1"


class TokenAuthDependency:
    """Validate bearer tokens as a FastAPI dependency."""

    async def __call__(self, request: Request) -> None:
        from fastapi import HTTPException

        from rune.api.local_auth_guard import (
            is_localhost_request,
            is_trusted_local_bypass_request,
        )

        auth_header: str = request.headers.get("authorization", "")

        # Allow localhost bypass only after origin, referer and sec-fetch-site checks pass.
        if not auth_header:
            host = request.client.host if request.client else ""
            if not token_required() and is_localhost_request(host):
                # Extract port from the Host header or server scope
                server_port = request.scope.get("server", (None, 0))[1] or 0
                headers = {k.decode(): v.decode() for k, v in request.scope.get("headers", [])}
                if is_trusted_local_bypass_request(headers, server_port):
                    return
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header",
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Use: Bearer <token>",
            )

        if not verify_token(token):
            raise HTTPException(
                status_code=403,
                detail="Invalid or expired token",
            )
