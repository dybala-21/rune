"""Why a hosted machine can't reach the open web, as its connector broker reports it.

Once a connector has sent a hosted machine its owner's private data, the host closes the open web
to it (rune-egress refuses every site the owner hasn't allowed), so an instruction hidden in a
page or an email can't send that data away. From inside the machine that only looks like a
connection that closed or a 403. When a network tool fails here, ask the broker; if that is why,
say so in the result, with where the owner can allow the site, so the agent tells the owner instead
of trying other ways round it.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Tools whose failure may be the closed web. Shell commands can reach the web too (curl).
_NETWORK_PREFIXES = ("web_", "browser_")
_NETWORK_TOOLS = frozenset({"bash_execute"})
# One question per few seconds is enough: a run that hits the closed web tends to hit it repeatedly.
_CACHE_SECONDS = 5.0
_cache: tuple[float, dict[str, Any] | None] = (0.0, None)


def touches_network(tool: str) -> bool:
    return tool.startswith(_NETWORK_PREFIXES) or tool in _NETWORK_TOOLS


async def closed() -> dict[str, Any] | None:
    """The broker's answer when the web is closed to this machine, else None (also when unsure)."""
    global _cache
    from rune.connectors.client import call_broker, hosted_approval

    if not hosted_approval():
        return None
    at, status = _cache
    if time.monotonic() - at < _CACHE_SECONDS:
        return status
    try:
        answer = await call_broker("status", {})
        status = answer if answer.get("closed") is True else None
    except Exception as exc:
        # An older broker has no status; the result then stays as it was.
        log.debug("web_access_status_unavailable", error=type(exc).__name__)
        status = None
    _cache = (time.monotonic(), status)
    return status


async def explain(result: CapabilityResult) -> CapabilityResult:
    """Say in a failed network result that the web is closed to this machine, when it is."""
    status = await closed()
    if status is None:
        return result
    url = str(status.get("approval_url") or "")
    where = f" at {url}" if url.startswith("https://") else " on the approval page"
    note = ("The open web is closed to this machine: a connector has sent it the owner's private data, so it "
            "reaches only sites the owner allows. Ask the owner to allow the site, or open browsing for an hour,"
            f"{where}. Don't look for another way round it.")
    error = f"{result.error}\n\n{note}" if result.error else note
    metadata = {**(result.metadata or {}), "web_access": "closed", "approval_url": url}
    return replace(result, error=error, metadata=metadata)
