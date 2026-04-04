"""Browser extended capabilities — batch, workflow, profile.

Split from browser.py. These higher-level orchestration tools
delegate to the core browser capabilities via the registry.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schemas
class BrowserBatchParams(BaseModel):
    """Parameters for batch browser operations."""
    actions: list[dict[str, Any]] = Field(description="List of browser actions to execute in sequence")


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
async def browser_batch(params: BrowserBatchParams) -> CapabilityResult:
    """Execute multiple browser actions in a single batch."""
    from rune.capabilities.registry import get_capability_registry

    reg = get_capability_registry()
    results: list[str] = []
    successes: list[bool] = []
    for i, action in enumerate(params.actions):
        action_type = action.get("type", "")
        try:
            result = await reg.execute(f"browser_{action_type}", action.get("params", {}))
            successes.append(result.success)
            results.append(f"Action {i+1} ({action_type}): {'OK' if result.success else result.error}")
        except Exception as exc:
            successes.append(False)
            results.append(f"Action {i+1} ({action_type}): Error — {exc}")
    return CapabilityResult(success=all(successes), output="\n".join(results))


async def browser_workflow(params: BrowserWorkflowParams) -> CapabilityResult:
    """Execute a multi-step browser workflow."""
    from rune.capabilities.registry import get_capability_registry

    reg = get_capability_registry()
    results: list[str] = []
    successes: list[bool] = []
    for i, step in enumerate(params.steps):
        step_type = step.get("action", "")
        try:
            result = await reg.execute(f"browser_{step_type}", step.get("params", {}))
            successes.append(result.success)
            results.append(f"Step {i+1} ({step_type}): {'OK' if result.success else result.error}")
            if not result.success and not step.get("continue_on_error", False):
                break
        except Exception as exc:
            successes.append(False)
            results.append(f"Step {i+1} ({step_type}): Error — {exc}")
            break
    return CapabilityResult(
        success=all(successes),
        output=f"Workflow '{params.name}':\n" + "\n".join(results),
    )


_browser_profiles: dict[str, dict] = {}


async def browser_profile(params: BrowserProfileParams) -> CapabilityResult:
    """Manage browser profiles for different contexts."""
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
