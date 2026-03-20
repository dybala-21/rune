"""Capability type definitions for RUNE.

Ported from src/capabilities/ - base interfaces for all tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from rune.types import CapabilityResult, Domain, RiskLevel


@dataclass(slots=True)
class CapabilityDefinition:
    """Metadata for a registered capability (tool)."""
    name: str
    description: str
    domain: Domain = Domain.GENERAL
    risk_level: RiskLevel = RiskLevel.LOW
    group: str = ""  # e.g. "read", "write", "runtime", "browser"
    parameters_model: type[BaseModel] | None = None
    raw_json_schema: dict[str, Any] | None = None  # For MCP tools without Pydantic model
    execute: Callable[..., Awaitable[CapabilityResult]] | None = None


# Groups for policy-based access control
TOOL_GROUPS: dict[str, list[str]] = {
    "read": ["file_read", "file_list", "file_search", "code_analyze",
             "code_find_def", "code_find_refs", "code_impact", "project_map"],
    "write": ["file_write", "file_edit", "file_delete", "credential_save",
              "skill_create", "skill_promote", "memory_tune"],
    "runtime": ["bash_execute"],
    "browser": ["browser_navigate", "browser_observe", "browser_act",
                 "browser_batch", "browser_extract", "browser_find",
                 "browser_workflow", "browser_screenshot", "browser_profile"],
    "web": ["web_search", "web_fetch", "service_connect", "service_status",
            "service_list", "service_disconnect", "service_reconnect"],
    "schedule": ["cron_create", "cron_list", "cron_delete", "cron_update"],
    "safe": ["think", "ask_user", "memory_search", "memory_save",
             "safety_tune", "task_create", "task_update", "task_list"],
    "delegate": ["delegate_task", "delegate_orchestrate"],
    "service": ["managed_service_status", "managed_service_stop"],
}


# Policy Profiles
# Ported from src/capabilities/types.ts - DEFAULT_PROFILES mapping.
# Each profile controls which tool groups are enabled, bash/file-write
# permissions, and the maximum risk level that can run without approval.

POLICY_PROFILES: dict[str, dict[str, Any]] = {
    "safe": {
        "description": "Read-only, no execution",
        "allowed_groups": ["safe", "read"],
        "allow_bash": False,
        "allow_file_write": False,
        "max_risk_level": "low",
    },
    "standard": {
        "description": "Standard development with approval gates",
        "allowed_groups": ["safe", "read", "write", "schedule"],
        "allow_bash": True,
        "allow_file_write": True,
        "max_risk_level": "medium",
        "require_approval": ["file_delete"],
    },
    "developer": {
        "description": "Full development access",
        "allowed_groups": ["safe", "read", "write", "schedule", "web", "browser", "runtime"],
        "allow_bash": True,
        "allow_file_write": True,
        "max_risk_level": "high",
        "deny": [],
    },
    "rune": {
        "description": "RUNE internal -- all tools with safety checks",
        "allowed_groups": [
            "safe", "read", "write", "schedule", "web", "browser",
            "delegate", "runtime", "service",
        ],
        "allow_bash": True,
        "allow_file_write": True,
        "max_risk_level": "high",
        "require_approval": ["file_delete", "managed_service_stop"],
    },
    "full": {
        "description": "Unrestricted -- all capabilities enabled",
        "allowed_groups": list(TOOL_GROUPS.keys()),
        "allow_bash": True,
        "allow_file_write": True,
        "max_risk_level": "critical",
    },
}


def get_allowed_tools(profile_name: str) -> list[str]:
    """Resolve a policy profile to a flat list of individual tool names.

    Expands the ``allowed_groups`` of the named profile through
    :data:`TOOL_GROUPS` and returns a deduplicated, sorted list.

    Raises ``KeyError`` if *profile_name* is not a known profile.
    """
    profile = POLICY_PROFILES[profile_name]
    tools: list[str] = []
    seen: set[str] = set()
    for group in profile["allowed_groups"]:
        for tool in TOOL_GROUPS.get(group, []):
            if tool not in seen:
                seen.add(tool)
                tools.append(tool)
    return sorted(tools)


def resolve_policy_profile(profile_name: str) -> dict[str, Any]:
    """Return the full profile dict with ``allowed_tools`` pre-resolved.

    The returned dict is a **copy** so callers can mutate it freely.

    Raises ``KeyError`` if *profile_name* is not a known profile.
    """
    profile = POLICY_PROFILES[profile_name]
    resolved = dict(profile)
    resolved["allowed_tools"] = get_allowed_tools(profile_name)
    return resolved
