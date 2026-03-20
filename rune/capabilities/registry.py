"""Capability registry for RUNE.

Ported from src/capabilities/index.ts - singleton registry with
policy-based access control and pattern matching.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from rune.capabilities.types import TOOL_GROUPS, CapabilityDefinition
from rune.types import CapabilityResult, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


class CapabilityRegistry:
    """Central registry for all agent capabilities (tools)."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityDefinition] = {}
        self._denied_patterns: list[str] = []
        self._require_approval_patterns: list[str] = []

    def register(self, cap: CapabilityDefinition) -> None:
        """Register a capability."""
        self._capabilities[cap.name] = cap
        log.debug("capability_registered", name=cap.name, domain=cap.domain)

    def get(self, name: str) -> CapabilityDefinition | None:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def list_all(self) -> list[CapabilityDefinition]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def list_names(self) -> list[str]:
        """List all capability names."""
        return list(self._capabilities.keys())

    def get_by_group(self, group: str) -> list[CapabilityDefinition]:
        """Get capabilities belonging to a group."""
        names = TOOL_GROUPS.get(group, [])
        return [self._capabilities[n] for n in names if n in self._capabilities]

    def is_allowed(self, name: str) -> bool:
        """Check if a capability is allowed by policy."""
        return all(not fnmatch(name, pattern) for pattern in self._denied_patterns)

    def requires_approval(self, name: str) -> bool:
        """Check if a capability requires approval."""
        cap = self._capabilities.get(name)
        if cap and cap.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return True
        return any(fnmatch(name, pattern) for pattern in self._require_approval_patterns)

    def set_denied_patterns(self, patterns: list[str]) -> None:
        self._denied_patterns = patterns

    def set_approval_patterns(self, patterns: list[str]) -> None:
        self._require_approval_patterns = patterns

    async def execute(self, name: str, params: dict[str, Any]) -> CapabilityResult:
        """Execute a capability by name."""
        cap = self._capabilities.get(name)
        if cap is None:
            return CapabilityResult(
                success=False, error=f"Unknown capability: {name}"
            )

        if not self.is_allowed(name):
            return CapabilityResult(
                success=False, error=f"Capability '{name}' is denied by policy"
            )

        if cap.execute is None:
            return CapabilityResult(
                success=False, error=f"Capability '{name}' has no execute function"
            )

        try:
            # Validate parameters if model is defined
            if cap.parameters_model is not None:
                validated = cap.parameters_model.model_validate(params)
                return await cap.execute(validated)
            return await cap.execute(params)
        except Exception as exc:
            return CapabilityResult(
                success=False, error=f"Capability '{name}' failed: {exc}"
            )


# Module singleton

_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
        _register_all_capabilities(_registry)
    return _registry


def _register_all_capabilities(registry: CapabilityRegistry) -> None:
    """Register all built-in capabilities."""
    from rune.capabilities.ask_user import register_ask_user_capability
    from rune.capabilities.bash import register_bash_capabilities
    from rune.capabilities.browser import register_browser_capabilities
    from rune.capabilities.code_intelligence import register_code_intelligence_capabilities
    from rune.capabilities.credential import register_credential_capabilities
    from rune.capabilities.cron import register_cron_capabilities
    from rune.capabilities.delegate import register_delegate_capabilities
    from rune.capabilities.file import register_file_capabilities
    from rune.capabilities.memory_capability import register_memory_capabilities
    from rune.capabilities.project import register_project_capabilities
    from rune.capabilities.safety_cap import register_safety_capabilities
    from rune.capabilities.service import register_service_capabilities
    from rune.capabilities.skill_ops import register_skill_ops_capabilities
    from rune.capabilities.task_ops import register_task_ops_capabilities
    from rune.capabilities.think import register_think_capabilities
    from rune.capabilities.web import register_web_capabilities

    register_file_capabilities(registry)
    register_bash_capabilities(registry)
    register_think_capabilities(registry)
    register_web_capabilities(registry)
    register_project_capabilities(registry)
    register_code_intelligence_capabilities(registry)
    register_memory_capabilities(registry)
    register_delegate_capabilities(registry)
    register_cron_capabilities(registry)
    register_task_ops_capabilities(registry)
    register_credential_capabilities(registry)
    register_skill_ops_capabilities(registry)
    register_browser_capabilities(registry)
    register_ask_user_capability(registry)
    register_service_capabilities(registry)
    register_safety_capabilities(registry)
