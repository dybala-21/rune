"""Approval manager for RUNE.

Ported 1:1 from src/safety/approval-manager.ts - session-cached approval,
command grouping, auto-approve logic, and config persistence.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

ApprovalResolutionSource = Literal[
    "prompt", "prompt_always",
    "auto_approve", "auto_approve_allowlist", "auto_approve_sandbox",
    "auto_deny", "fail_closed",
]

ApprovalPromptDecision = bool | Literal["approve_once", "approve_always", "deny"]


@dataclass(slots=True)
class ApprovalRequestInfo:
    command: str = ""
    risk_level: str = "medium"
    sandbox_planned: bool = False
    resolution_source: ApprovalResolutionSource | None = None
    reason: str = ""
    suggestions: list[str] = field(default_factory=list)
    user_guidance: str = ""


ApprovalPromptHandler = Callable[[ApprovalRequestInfo], Awaitable[ApprovalPromptDecision]]
ApprovalResolver = Callable[[ApprovalRequestInfo], Awaitable[bool]]

NormalizedRisk = Literal["safe", "low", "medium", "high", "critical"]

_KNOWN_RISKS: set[NormalizedRisk] = {"safe", "low", "medium", "high", "critical"}


@dataclass(slots=True)
class _CachedDecision:
    approved: bool
    source: ApprovalResolutionSource
    at: float


# Approval Config

@dataclass(slots=True)
class ApprovalConfig:
    profile: Literal["general", "developer", "automation"] = "general"
    auto_approve_low_risk: bool = True
    auto_approve_sandboxed_high_risk: bool = False
    always_allow_command_groups: list[str] = field(default_factory=list)
    require_explicit_for: list[str] = field(default_factory=list)
    session_cache_enabled: bool = True
    session_cache_max_entries: int = 200
    session_cache_include_denied: bool = False


# Helpers (ported 1:1)

def _normalize_risk_level(risk_level: str) -> NormalizedRisk:
    normalized = str(risk_level or "").strip().lower()
    if normalized in _KNOWN_RISKS:
        return normalized  # type: ignore[return-value]
    return "high"


def _strip_env_prefix(command: str) -> str:
    import re
    return re.sub(r"^(\w+=\S+\s+)+", "", command.strip())


_GROUPED_EXECUTABLES = {
    "git", "npm", "pnpm", "yarn", "docker",
    "python", "python3", "node", "npx", "uv", "cargo", "go",
}


def _command_group(command: str) -> str:
    stripped = _strip_env_prefix(command)
    tokens = stripped.split()
    if not tokens:
        return "(empty)"

    executable = (tokens[0].rsplit("/", 1)[-1]).lower()
    sub = tokens[1].lower() if len(tokens) > 1 else None

    if executable in _GROUPED_EXECUTABLES:
        if not sub or sub.startswith("-"):
            return executable
        return f"{executable}:{sub}"
    return executable


def _matches_explicit_pattern(command: str, group: str, patterns: list[str]) -> bool:
    lowered = command.lower()
    for raw in patterns:
        pattern = raw.strip().lower()
        if not pattern:
            continue
        if group == pattern or group.startswith(f"{pattern}:") or pattern in lowered:
            return True
    return False


def _normalize_prompt_decision(decision: ApprovalPromptDecision) -> tuple[bool, bool]:
    """Returns (approved, always)."""
    if decision == "approve_always":
        return True, True
    if decision == "approve_once":
        return True, False
    if decision == "deny":
        return False, False
    if isinstance(decision, bool):
        return decision, False
    return False, False


# Config persistence

async def _persist_always_allow_group(group: str) -> None:
    """Write always-allow group to ~/.rune/config.yaml."""
    config_dir = Path.home() / ".rune"
    config_path = config_dir / "config.yaml"

    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True

        parsed: dict = {}
        if config_path.is_file():
            loaded = yaml.load(config_path)
            if isinstance(loaded, dict):
                parsed = loaded

        approval = parsed.get("approval", {})
        if not isinstance(approval, dict):
            approval = {}

        groups = approval.get("alwaysAllowCommandGroups", [])
        if not isinstance(groups, list):
            groups = []

        if group not in groups:
            groups.append(group)

        approval["alwaysAllowCommandGroups"] = groups
        parsed["approval"] = approval

        config_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(parsed, config_path)

    except Exception as exc:
        log.warning("persist_always_allow_failed", group=group, error=str(exc))


# Session Approval Resolver

def create_session_approval_resolver(
    config: ApprovalConfig,
    prompt_handler: ApprovalPromptHandler | None = None,
    *,
    persist_fn: Callable[[str], Awaitable[None]] | None = None,
) -> ApprovalResolver:
    """Create a session-scoped approval resolver.

    Returns an async callable that resolves approval decisions with caching.
    """
    cache: dict[str, _CachedDecision] = {}
    persist = persist_fn or _persist_always_allow_group

    async def resolve(info: ApprovalRequestInfo) -> bool:
        risk = _normalize_risk_level(info.risk_level)
        group = _command_group(info.command)
        cache_key = f"{group}|{risk}"
        always_allow = list(config.always_allow_command_groups)
        should_force_prompt = _matches_explicit_pattern(
            info.command, group, config.require_explicit_for,
        )

        # Check session cache
        if config.session_cache_enabled:
            cached = cache.get(cache_key)
            if cached is not None:
                info.resolution_source = cached.source
                return cached.approved

        approved: bool
        source: ApprovalResolutionSource

        is_allowlisted = _matches_explicit_pattern(info.command, group, always_allow)

        should_auto_by_profile = (
            config.profile == "general" and risk in ("safe", "low", "medium")
        )
        should_auto_by_flag = (
            config.auto_approve_low_risk and risk in ("safe", "low")
        )
        should_auto_sandbox_high = (
            config.auto_approve_sandboxed_high_risk
            and config.profile == "general"
            and risk == "high"
            and info.sandbox_planned
        )

        # Decision tree
        if is_allowlisted:
            approved = True
            source = "auto_approve_allowlist"
        elif not should_force_prompt and (
            should_auto_by_profile or should_auto_by_flag or should_auto_sandbox_high
        ):
            approved = True
            source = "auto_approve_sandbox" if should_auto_sandbox_high else "auto_approve"
        elif config.profile == "automation" and risk in ("high", "critical"):
            approved = False
            source = "auto_deny"
        elif prompt_handler is None:
            approved = False
            source = "fail_closed"
        else:
            prompt_decision = await prompt_handler(info)
            approved_result, always = _normalize_prompt_decision(prompt_decision)
            approved = approved_result
            source = "prompt_always" if always else "prompt"

            if always:
                if group not in always_allow:
                    always_allow.append(group)
                with contextlib.suppress(Exception):
                    await persist(group)

        # Cache
        if config.session_cache_enabled:
            if approved or config.session_cache_include_denied:
                cache[cache_key] = _CachedDecision(
                    approved=approved, source=source, at=time.time(),
                )
                # Enforce max entries
                while len(cache) > config.session_cache_max_entries:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

        info.resolution_source = source
        return approved

    return resolve
