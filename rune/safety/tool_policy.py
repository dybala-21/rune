"""Shared approval policy for model tools and direct capability calls."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, NamedTuple

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


# File-mutating capability names

_FILE_MUTATING_CAPABILITIES = frozenset({
    "file_write", "file_edit", "file_delete",
    "document_create", "document_bundle", "document_bundle_update",
})

# Default pattern for detecting MCP write operations
_MCP_WRITE_PATTERN = re.compile(
    r"create|update|delete|remove|send|post|put|patch|write|insert|modify|edit|add|move|archive",
    re.IGNORECASE,
)

def is_mcp_write_operation(cap_name: str) -> bool:
    """Recognize write verbs in the tool-name portion of an MCP capability."""
    if not cap_name.startswith("mcp."):
        return False
    parts = cap_name.split(".")
    if len(parts) < 3:
        return False
    tool_name = ".".join(parts[2:])
    return bool(_MCP_WRITE_PATTERN.search(tool_name))


# Network approval helper

# Strict mode also asks before network reads and browser navigation.
_NETWORK_READ_CAPS = frozenset({
    "web_search", "browser_navigate", "browser_observe", "browser_extract",
    "browser_find", "browser_screenshot", "browser_discover_apis",
})
# Browser input needs approval in strict mode: a click may submit a form.
_NETWORK_ACT_CAPS = frozenset({
    "browser_act", "browser_batch", "browser_workflow",
})


_APPROVAL_MODES = ("bypass", "standard", "strict")


def approval_mode() -> str:
    """Resolve approval settings, keeping hosted workers at least on standard."""
    from rune.cloud.boundary import hosted

    def effective(mode: str) -> str:
        if mode not in _APPROVAL_MODES or (hosted() and mode == "bypass"):
            return "standard"
        return mode

    env = os.environ.get("RUNE_APPROVAL_MODE", "").strip().lower()
    if env in _APPROVAL_MODES:
        return effective(env)
    try:
        from rune.config.loader import get_config

        mode = str(getattr(get_config().approval, "mode", "standard")).strip().lower()
        return effective(mode)
    except Exception as exc:
        log.debug("approval_mode_config_read_failed", error=str(exc)[:100])
        return "standard"


def _network_writes_possible() -> bool:
    """Check whether a non-GET web_fetch is enabled and can reach the network."""
    return os.environ.get("RUNE_HYBRID_API", "0") == "1"


def _url_host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc or url[:60]
    except Exception:
        return url[:60]


class _NetworkApproval(NamedTuple):
    display: str
    reason: str
    cache_key: str
    is_write: bool = False
    single_use: bool = False


def _network_approval_request(
    cap_name: str, params: dict[str, Any]
) -> _NetworkApproval | None:
    """Return (display, reason, cache key) when the network call needs approval."""
    mode = approval_mode()
    if mode == "bypass":
        return None

    if cap_name == "connector_request":
        method = str(params.get("method") or "GET")
        target = str(params.get("origin", "")) + str(params.get("path", ""))
        return _NetworkApproval(f"{method} {target[:160]}",
                                f"Connected API: {method} {target[:160]}", "",
                                is_write=method != "GET", single_use=True)

    if cap_name == "web_fetch":
        method = str(params.get("method") or "GET").upper()
        url = str(params.get("url") or "")
        host = _url_host(url)
        if method != "GET" and _network_writes_possible():
            return _NetworkApproval(
                f"{method} {url[:120]}",
                f"Network write to {host} — this cannot be undone",
                f"{method}|{host}",
                is_write=True,
            )
        if mode == "strict":
            return _NetworkApproval(f"GET {url[:120]}", f"Fetch {host}", f"GET|{host}")
        return None

    if mode != "strict":
        return None

    if cap_name in _NETWORK_READ_CAPS:
        target = str(params.get("url") or params.get("query") or "")
        return _NetworkApproval(
            f"{cap_name} {target[:120]}".strip(),
            f"Network read: {cap_name}",
            f"{cap_name}|{_url_host(target) if target.startswith('http') else target[:60]}",
        )

    if cap_name in _NETWORK_ACT_CAPS:
        action = str(params.get("action") or "")
        selector = str(params.get("selector") or "")
        return _NetworkApproval(
            f"{cap_name} {action} {selector}".strip(),
            f"Browser interaction: {action or cap_name}",
            f"{cap_name}|{action}",
            is_write=True,
        )

    return None


# Guardian validation helper

@dataclass(slots=True)
class _GuardianResult:
    """Internal result from Guardian validation."""
    blocked: bool = False
    requires_approval: bool = False
    reason: str = ""


def _capability_asked_for_approval(result: CapabilityResult) -> bool:
    """Check whether a refused capability call can proceed with user approval."""
    if result.success:
        return False
    meta = result.metadata
    return bool(isinstance(meta, dict) and meta.get("requires_approval"))


def _normalise_cap_name(name: str) -> str:
    """file.delete / file-delete / file_delete are the same tool to an operator."""
    return name.strip().lower().replace(".", "_").replace("-", "_")


def _explicit_approval_patterns() -> list[str]:
    """Capabilities the operator marked as always-prompt, normalised."""
    try:
        from rune.config import get_config

        raw = get_config().approval.require_explicit_for or []
    except Exception as exc:
        log.debug("require_explicit_for_read_failed", error=str(exc))
        raise RuntimeError("Approval policy is unavailable") from exc
    return [_normalise_cap_name(str(x)) for x in raw if str(x).strip()]


def _requires_explicit_approval(cap_name: str) -> bool:
    """Whether ``approval.requireExplicitFor`` names this capability."""
    target = _normalise_cap_name(cap_name)
    return any(fnmatch(target, pat) for pat in _explicit_approval_patterns())


def _validate_with_guardian(cap_name: str, params: dict[str, Any]) -> _GuardianResult:
    """Return the Guardian decision: blocked, approval required, or both false for allowed."""
    try:
        from rune.cloud.boundary import check_tool
        from rune.safety.guardian import get_guardian

        if reason := check_tool(cap_name, params):
            return _GuardianResult(blocked=True, reason=reason)
        guardian = get_guardian()

        if cap_name == "bash_execute":
            command = params.get("command", "")
            result = guardian.validate(command, cwd=params.get("cwd"))
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked bash: {result.reason}")
            if result.requires_approval:
                return _GuardianResult(requires_approval=True, reason=result.reason)

        elif cap_name in _FILE_MUTATING_CAPABILITIES:
            file_path = params.get("file_path") or params.get("path") or params.get("directory", "")
            result = guardian.validate_file_path(file_path)
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked file write: {result.reason}")
            write_approval = result.requires_approval
            write_reason = result.reason
            if cap_name == "document_bundle":
                result = guardian.validate_file_read_path(params.get("source_path", ""))
                if not result.allowed:
                    return _GuardianResult(blocked=True, reason=f"Guardian blocked source read: {result.reason}")
            if write_approval:
                return _GuardianResult(requires_approval=True, reason=write_reason)

        elif cap_name in ("file_read", "document_read", "document_preview", "document_bundle_inspect"):
            file_path = params.get("file_path") or params.get("path") or params.get("directory", "")
            result = guardian.validate_file_read_path(file_path)
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked file read: {result.reason}")

        if _requires_explicit_approval(cap_name):
            return _GuardianResult(
                requires_approval=True,
                reason=f"{cap_name} requires explicit approval (approval.requireExplicitFor)",
            )

    except Exception as exc:
        log.error("guardian_validation_error", error=str(exc))
        # Fail closed: block if Guardian itself errors
        return _GuardianResult(blocked=True, reason=f"Guardian validation error (fail-closed): {exc}")

    return _GuardianResult()
