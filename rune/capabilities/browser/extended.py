"""Browser extended capabilities — batch, workflow, profile.

Split from browser.py. These higher-level orchestration tools
delegate to the core browser capabilities via the registry.
"""

from __future__ import annotations

from typing import Any, ClassVar, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rune.capabilities.browser.session import browser_operation, current_session
from rune.capabilities.browser.steps import BrowserStep
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


def _batch_schema(schema: dict) -> None:
    # Vertex accepts anyOf; keep the discriminator for local validation only.
    items = schema["properties"]["actions"]["items"]
    items["anyOf"] = items.pop("oneOf")
    items.pop("discriminator", None)


class BrowserBatchParams(BaseModel):
    """Parameters for batch browser operations."""
    model_config = ConfigDict(json_schema_extra=_batch_schema)
    validation_hint: ClassVar[str] = (
        "No steps in this batch executed. Keep earlier successful work; do not resend earlier batches. "
        "Use one standalone browser_act for the next unfinished action to recover. "
        'Batch shape: {"actions":[{"type":"act","params":{"action":"click",'
        '"selector":"CURRENT_REF"}}]}. Put checks inside params, each with its own result selector.'
    )
    actions: list[BrowserStep] = Field(description=(
        "Browser tool calls in order, each with type and params. Stops after a tool or result-check failure. "
        'Example: {"type":"act","params":{"action":"click","selector":"CURRENT_REF"}}. '
        "For a known form, check all requested values on the last field edit before the save step. "
        "Refs remain valid only while the observed node, name and page are unchanged. "
        "After navigation or replacing controls, read the returned refs before continuing."
    ), min_length=1, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def wrap_action_steps(cls, values):
        from rune.capabilities.browser.capabilities import BrowserActParams

        if not isinstance(values, dict) or not isinstance(values.get("actions"), list):
            return values
        fields = set(BrowserActParams.model_fields)
        actions = get_args(BrowserActParams.model_fields["action"].annotation)
        def normalize(step):
            if not isinstance(step, dict) or "params" in step:
                return step
            tag = step.get("type")
            if tag == "act" and "action" in step:
                step = {key: value for key, value in step.items() if key != "type"}
            elif tag in actions and "action" not in step:
                step = {**{key: value for key, value in step.items() if key != "type"}, "action": tag}
            if "action" in step and set(step) <= fields:
                return {"type": "act", "params": step}
            return step
        return {**values, "actions": [normalize(step) for step in values["actions"]]}


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
    evidence: list[dict] = []
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
        evidence.append({"tool": f"browser_{step_type}", "success": result.success,
                         "metadata": result.metadata or {}, "error": result.error})
        checks = (result.metadata or {}).get("result_checks")
        check_failed = isinstance(checks, dict) and checks.get("passed") is not True
        changed |= step_type in {"act", "navigate", "open"} and result.success
        if not result.success or check_failed:
            error = result.error or "A result check failed. Later steps were not executed. Read the returned state; do not repeat completed actions."
            results.extend(f"Step {j+1}: Not executed because an earlier step failed."
                           for j in range(i + 1, len(steps)))
            return CapabilityResult(success=False, error=error, output="\n".join(results),
                                    metadata={"browser_steps": evidence, "verification_failed": check_failed and result.success,
                                              "action_status": "unknown" if changed else
                                              (result.metadata or {}).get("action_status", "unknown")})
        if step_type == "screenshot" and result.metadata.get("path"):
            image_paths.append(result.metadata["path"])
    return CapabilityResult(success=True, output="\n".join(results),
                            metadata={"image_paths": image_paths[-2:], "browser_steps": evidence})


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
