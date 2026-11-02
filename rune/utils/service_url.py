"""Validate operator-configured service URLs (LLM gateway, remote connector broker)."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit


def service_url(value: str, *, allow_path: bool) -> str:
    """Return ``value`` without a trailing slash, or raise ValueError.

    HTTPS is required, except plain HTTP to a private IP literal (a host-only link).
    Credentials, queries, fragments, bad ports, link-local or unspecified addresses
    (e.g. a cloud metadata endpoint) are refused, because prompts and keys go there.
    """
    if not value or re.search(r"[\s?#\\]", value):
        raise ValueError("the URL must not contain whitespace, a query or a fragment")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"the URL is malformed ({exc})") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("the URL must be http(s)://host[:port] without credentials")
    if port == 0:
        raise ValueError("the URL port is invalid")
    if not allow_path and parsed.path not in {"", "/"}:
        raise ValueError("the URL must not have a path")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is not None and (address.is_link_local or address.is_unspecified or address.is_multicast):
        raise ValueError("the URL must not point at a link-local, unspecified or multicast address")
    if scheme == "http" and (address is None or not address.is_private):
        raise ValueError("plain HTTP is only allowed to a private IP address")
    return value.rstrip("/")
