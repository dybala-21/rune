"""Local machine auth guard.

Ported from src/api/local-auth-guard.ts - validates that requests come
from localhost and are not malicious cross-origin browser requests (CSRF
protection for the localhost admin bypass).
"""

from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "::ffff:127.0.0.1"})


def _is_loopback_host(raw_host: str | None) -> bool:
    """Check if a host string resolves to a loopback address."""
    if not raw_host:
        return False
    normalized = raw_host.strip().lower()
    # Strip port if present (e.g. "127.0.0.1:8000")
    if ":" in normalized and not normalized.startswith("["):
        # Could be host:port - check if it's IPv6 first
        parts = normalized.rsplit(":", 1)
        if parts[1].isdigit():
            normalized = parts[0]
    return normalized in LOOPBACK_HOSTS


def _is_trusted_browser_source(raw_url: str | None, expected_port: int) -> bool:
    """Verify that a browser Origin/Referer header points to a trusted local source."""
    if not raw_url:
        return True  # No Origin/Referer => non-browser request, allow
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return False

    if not _is_loopback_host(parsed.hostname):
        return False

    return not (parsed.port is not None and parsed.port != expected_port)


def is_trusted_local_bypass_request(
    headers: dict[str, str],
    expected_port: int,
) -> bool:
    """Determine if a request qualifies for the localhost admin bypass.

    Rules:
    - If ``sec-fetch-site`` is ``cross-site``, deny immediately.
    - CLI/curl calls (no Origin/Referer) are allowed.
    - Browser calls must have loopback Origin and Referer.

    Args:
        headers: Mapping of header names (lowercased) to values.
        expected_port: The port number the server is listening on.

    Returns:
        ``True`` if the request is trusted.
    """
    sec_fetch_site = (headers.get("sec-fetch-site") or "").strip().lower()
    if sec_fetch_site and sec_fetch_site not in ("same-origin", "same-site", "none"):
        return False

    origin = headers.get("origin")
    if not _is_trusted_browser_source(origin, expected_port):
        return False

    referer = headers.get("referer")
    return _is_trusted_browser_source(referer, expected_port)


def is_localhost_request(client_host: str | None) -> bool:
    """Check if the client IP is a loopback address.

    Args:
        client_host: The client IP address from the request.

    Returns:
        ``True`` if the client is connecting from localhost.
    """
    return _is_loopback_host(client_host)
