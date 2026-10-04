"""Register capabilities and enforce their execution policies."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from pydantic import ValidationError

from rune.agent.timing import timed
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

    @timed("tool", name_arg=1)
    async def execute(self, name: str, params: dict[str, Any]) -> CapabilityResult:
        """Execute a capability by name."""
        from rune.computer.session import current_desktop
        if desktop := current_desktop():
            blocker = desktop.tool_blocker(name, params)
            if blocker:
                return CapabilityResult(success=False, error=blocker, metadata={"action_status": "not_executed"})
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
                normalized = validated.model_dump(mode="json", by_alias=True)
            else:
                from copy import deepcopy
                validated, normalized = deepcopy(params), deepcopy(params)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in exc.errors(include_input=False, include_url=False)[:3]
            )
            return CapabilityResult(success=False, error=f"Invalid arguments for {name}: {details[:1000]}",
                                    metadata={"action_status": "not_executed"})
        except Exception as exc:
            return CapabilityResult(success=False, error=f"Capability '{name}' failed before execution: {exc}",
                                    metadata={"action_status": "not_executed"})
        try:
            from rune.agent.execution_journal import active_journal
            from rune.agent.run_control import dispatch_scope
            from rune.safety.approval_context import (
                approval_granted,
                approval_required,
                approval_revisions,
                consume_approval,
            )
            from rune.safety.resource_locks import capability_access
            from rune.safety.tool_policy import (
                _network_approval_request,
                _validate_with_guardian,
                approval_mode,
                is_mcp_write_operation,
            )

            guard = _validate_with_guardian(name, normalized)
            if guard.blocked:
                return CapabilityResult(success=False, error=guard.reason,
                                        metadata={"action_status": "not_executed"})
            approved = consume_approval(name, normalized)
            revisions = approval_revisions() if approved else {}
            bypass = approval_mode() == "bypass"
            network = _network_approval_request(name, normalized)
            # Shell risk is assessed per command by Guardian and the executor.
            pinned = any(fnmatch(name, pattern) for pattern in self._require_approval_patterns)
            needs_approval = (guard.requires_approval or pinned or network is not None
                              or is_mcp_write_operation(name)
                              or name != "bash_execute" and self.requires_approval(name))
            if needs_approval and not (approved or bypass):
                reason = guard.reason or (network.reason if network else f"{name} requires approval")
                return CapabilityResult(success=False, error=reason, metadata={
                    "action_status": "not_executed", "requires_approval": True, "reason": reason,
                })

            journal = active_journal()
            async def invoke():
                from rune.agent.execution_journal import RecoveryBlocked
                from rune.safety.approval_request import check_revisions
                try:
                    check_revisions(revisions)
                except RecoveryBlocked as exc:
                    return CapabilityResult(success=False, error=str(exc), metadata={"action_status": "not_executed"})
                return await cap.execute(validated)

            async with dispatch_scope():
                # Nested capability calls must obtain their own scoped approval.
                with (approval_granted() if approved or bypass else approval_required()), capability_access(name, normalized):
                    if journal is not None:
                        return await journal.execute(name, normalized, invoke)
                    return await invoke()
        except Exception as exc:
            from rune.safety.resource_locks import ResourceBusy
            if isinstance(exc, ResourceBusy):
                return CapabilityResult(success=False, error=str(exc), metadata={
                    "action_status": "not_executed", "resource_busy": True,
                })
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
        _apply_disabled_capabilities(_registry)
    return _registry


def _apply_disabled_capabilities(registry: CapabilityRegistry) -> None:
    """Apply comma-separated exact names or prefix* patterns from RUNE_DISABLED_CAPABILITIES."""
    import os

    from rune.cloud.boundary import check_tool, hosted

    if hosted():
        for name in registry.list_names():
            if check_tool(name, {}):
                registry._capabilities.pop(name, None)

    raw = os.environ.get("RUNE_DISABLED_CAPABILITIES", "").strip()
    if not raw:
        return
    patterns = [p.strip() for p in raw.split(",") if p.strip()]
    removed = []
    for name in registry.list_names():
        for pat in patterns:
            matched = name.startswith(pat[:-1]) if pat.endswith("*") else name == pat
            if matched:
                registry._capabilities.pop(name, None)
                removed.append(name)
                break
    if removed:
        log.info("capabilities_disabled_by_env", names=removed)


def _register_all_capabilities(registry: CapabilityRegistry) -> None:
    """Register all built-in capabilities."""
    from rune.capabilities.ask_user import register_ask_user_capability
    from rune.capabilities.bash import register_bash_capabilities
    from rune.capabilities.blocked import register_blocked_capability
    from rune.capabilities.browser import register_browser_capabilities
    from rune.capabilities.code_intelligence import register_code_intelligence_capabilities
    from rune.capabilities.connector import register_connector_capabilities
    from rune.capabilities.credential import register_credential_capabilities
    from rune.capabilities.cron import register_cron_capabilities
    from rune.capabilities.delegate import register_delegate_capabilities
    from rune.capabilities.document import register_document_capability
    from rune.capabilities.document_bundle import register_document_bundle_capability
    from rune.capabilities.file import register_file_capabilities
    from rune.capabilities.memory_capability import register_memory_capabilities
    from rune.capabilities.project import register_project_capabilities
    from rune.capabilities.safety_cap import register_safety_capabilities
    from rune.capabilities.service import register_service_capabilities
    from rune.capabilities.skill_ops import register_skill_ops_capabilities
    from rune.capabilities.table_checks import register_table_checks
    from rune.capabilities.task_ops import register_task_ops_capabilities
    from rune.capabilities.think import register_think_capabilities
    from rune.capabilities.web import register_web_capabilities
    from rune.computer.capabilities import register_desktop_capabilities

    register_file_capabilities(registry)
    register_document_capability(registry)
    register_document_bundle_capability(registry)
    register_table_checks(registry)
    register_bash_capabilities(registry)
    register_think_capabilities(registry)
    register_web_capabilities(registry)
    register_connector_capabilities(registry)
    register_project_capabilities(registry)
    register_code_intelligence_capabilities(registry)
    register_memory_capabilities(registry)
    register_delegate_capabilities(registry)
    register_cron_capabilities(registry)
    register_task_ops_capabilities(registry)
    register_credential_capabilities(registry)
    register_skill_ops_capabilities(registry)
    register_browser_capabilities(registry)
    register_desktop_capabilities(registry)
    register_ask_user_capability(registry)
    register_service_capabilities(registry)
    register_safety_capabilities(registry)
    register_blocked_capability(registry)
