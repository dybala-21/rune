"""Safety tuning capability for RUNE.

Ported from src/capabilities/safety.ts - applies safety preset
(conservative / balanced / developer) to the user config.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Parameter schema

class SafetyTuneParams(BaseModel):
    """Parameters for the safety.tune capability."""
    preset: Literal["conservative", "balanced", "developer"] = Field(
        description="Safety preset to apply (conservative / balanced / developer)",
    )


# Preset definitions

#: Preset -> safety-related config overrides.
_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "conservative": {
        "auto_approve": False,
        "max_risk_auto": "LOW",
        "require_approval_for_write": True,
        "sandbox_commands": True,
    },
    "balanced": {
        "auto_approve": False,
        "max_risk_auto": "MEDIUM",
        "require_approval_for_write": False,
        "sandbox_commands": True,
    },
    "developer": {
        "auto_approve": True,
        "max_risk_auto": "HIGH",
        "require_approval_for_write": False,
        "sandbox_commands": False,
    },
}


def _apply_safety_preset(preset: str) -> dict[str, Any]:
    """Return the config patch for the given preset name."""
    patch = _PRESET_OVERRIDES.get(preset)
    if patch is None:
        raise ValueError(f"Unknown safety preset: {preset}")
    return {"preset": preset, **patch}


# Capability implementation

async def safety_tune(params: SafetyTuneParams) -> CapabilityResult:
    """Apply a safety tuning preset."""
    try:
        patch = _apply_safety_preset(params.preset)

        # Persist to config
        from rune.config import get_config

        config = get_config()
        safety_cfg = getattr(config, "safety", None)
        if safety_cfg is not None:
            for key, value in patch.items():
                if hasattr(safety_cfg, key):
                    setattr(safety_cfg, key, value)

        from rune.utils.fast_serde import json_encode
        snapshot_str = json_encode(patch)

        return CapabilityResult(
            success=True,
            output=f'Updated safety tuning preset to "{params.preset}". Current: {snapshot_str}',
            metadata={
                "preset": params.preset,
                "safetyTuning": patch,
            },
        )
    except Exception as exc:
        err_msg = str(exc)
        log.error("safety_tune_failed", error=err_msg)
        return CapabilityResult(
            success=False,
            error=f"Safety tuning failed: {err_msg}",
        )


# Registration

def register_safety_capabilities(registry: CapabilityRegistry) -> None:
    """Register the safety.tune capability."""
    registry.register(CapabilityDefinition(
        name="safety_tune",
        description="Apply guarded safety execution preset (conservative / balanced / developer)",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.MEDIUM,
        group="safe",
        parameters_model=SafetyTuneParams,
        execute=safety_tune,
    ))
