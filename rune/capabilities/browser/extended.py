"""Browser extended capabilities — batch, workflow, profile.

Split from browser.py. These higher-level orchestration tools
delegate to the core browser capabilities via the registry.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rune.capabilities.browser.session import browser_operation, current_session
from rune.capabilities.browser.steps import BrowserStep
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schemas
class BrowserBatchParams(BaseModel):
    """Parameters for batch browser operations."""
    actions: list[BrowserStep] = Field(description=(
        "Browser tool calls in order, each with type and params. Stops after a failure. "
        "Refs remain valid only while the observed node, name and page are unchanged. "
        "After navigation or replacing controls, read the returned refs before continuing."
    ), min_length=1, max_length=20)


class BrowserWorkflowParams(BaseModel):
    """Parameters for browser workflow automation."""
    name: str = Field(description="Workflow name")
    steps: list[dict[str, Any]] = Field(description="Workflow steps to execute")
    timeout: int = Field(default=30000, description="Workflow timeout in ms")


class BrowserProfileParams(BaseModel):
    """Parameters for browser profile management."""
    action: str = Field(description="Profile action: create, load, delete")
    name: str = Field(description="Profile name")
    settings: dict[str, Any] = Field(default_factory=dict, description="Profile settings")


# Capability implementations
async def _run_steps(steps: list[dict[str, Any]], type_key: str) -> CapabilityResult:
    from rune.capabilities.registry import get_capability_registry

    reg = get_capability_registry()
    results: list[str] = []
    image_paths: list[str] = []
    changed = False
    allowed = {"navigate", "open", "observe", "act", "screenshot", "extract", "find", "discover_apis"}
    for i, step in enumerate(steps):
        step_type = step.get(type_key, "")
        try:
            if step_type not in allowed:
                raise ValueError(f"Unsupported browser step: {step_type}")
            result = await reg.execute(f"browser_{step_type}", step.get("params", {}))
        except Exception as exc:
            result = CapabilityResult(success=False, error=str(exc))
        results.append(f"Step {i+1} ({step_type}): {result.output if result.success else result.error}")
        if not result.success:
            results.extend(f"Step {j+1}: Not executed because an earlier step failed."
                           for j in range(i + 1, len(steps)))
            return CapabilityResult(success=False, error=result.error, output="\n".join(results),
                                    metadata={"action_status": "unknown" if changed else
                                              (result.metadata or {}).get("action_status", "unknown")})
        changed |= step_type in {"act", "navigate", "open"}
        if step_type == "screenshot" and result.metadata.get("path"):
            image_paths.append(result.metadata["path"])
    return CapabilityResult(success=True, output="\n".join(results), metadata={"image_paths": image_paths[-2:]})


@browser_operation
async def browser_batch(params: BrowserBatchParams) -> CapabilityResult:
    """Execute browser steps in order, stopping after the first failure."""
    return await _run_steps([step.model_dump(mode="json", by_alias=True) for step in params.actions], "type")


@browser_operation
async def browser_workflow(params: BrowserWorkflowParams) -> CapabilityResult:
    """Execute browser steps in order, stopping after the first failure."""
    result = await _run_steps(params.steps, "action")
    result.output = f"Workflow '{params.name}':\n{result.output}"
    return result


@browser_operation
async def browser_profile(params: BrowserProfileParams) -> CapabilityResult:
    """Manage browser profiles for different contexts."""
    _browser_profiles = current_session().profiles
    if params.action == "create":
        _browser_profiles[params.name] = {"settings": params.settings or {}}
        return CapabilityResult(success=True, output=f"Profile '{params.name}' created.")
    elif params.action == "delete":
        _browser_profiles.pop(params.name, None)
        return CapabilityResult(success=True, output=f"Profile '{params.name}' deleted.")
    elif params.action == "list":
        names = list(_browser_profiles.keys())
        return CapabilityResult(success=True, output=f"Profiles: {', '.join(names) or '(none)'}")
    elif params.action == "get":
        profile = _browser_profiles.get(params.name)
        if profile:
            return CapabilityResult(success=True, output=str(profile))
        return CapabilityResult(success=False, error=f"Profile '{params.name}' not found.")
    return CapabilityResult(success=False, error=f"Unknown action: {params.action}")
