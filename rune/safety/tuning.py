"""Safety parameter tuning and presets.

Ported from src/safety/tuning.ts - defines tuning presets (conservative,
balanced, developer) that map to rollout modes and auto-management settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SafetyTuningPreset = Literal["conservative", "balanced", "developer"]

# Types


@dataclass(slots=True)
class SafetyTuningSnapshot:
    """Current safety tuning state."""

    preset: SafetyTuningPreset | None
    rollout_mode: Literal["auto", "shadow", "balanced", "strict", "legacy"]
    auto_enabled: bool


@dataclass(slots=True)
class SafetyTuningPatch:
    """A patch to apply to safety tuning."""

    preset: SafetyTuningPreset


# Presets

@dataclass(slots=True, frozen=True)
class _PresetValues:
    rollout_mode: Literal["auto", "shadow", "balanced", "strict"]
    auto_enabled: bool


SAFETY_TUNING_PRESETS: dict[SafetyTuningPreset, _PresetValues] = {
    "conservative": _PresetValues(rollout_mode="strict", auto_enabled=False),
    "balanced": _PresetValues(rollout_mode="auto", auto_enabled=True),
    "developer": _PresetValues(rollout_mode="balanced", auto_enabled=False),
}


# Helpers

def _parse_preset(value: str | None) -> SafetyTuningPreset | None:
    """Parse a string into a valid preset name, or None."""
    normalized = (value or "").strip().lower()
    if normalized in ("conservative", "balanced", "developer"):
        return normalized  # type: ignore[return-value]
    return None


def _detect_preset(
    rollout_mode: str,
    auto_enabled: bool,
) -> SafetyTuningPreset | None:
    """Detect which preset matches the given configuration, if any."""
    for preset, values in SAFETY_TUNING_PRESETS.items():
        if values.rollout_mode == rollout_mode and values.auto_enabled == auto_enabled:
            return preset
    return None


# Public API

def resolve_safety_tuning(
    *,
    rollout_mode: str = "auto",
    auto_enabled: bool = True,
) -> SafetyTuningSnapshot:
    """Build a tuning snapshot from the current safety configuration.

    Parameters
    ----------
    rollout_mode:
        The active rollout mode string (e.g. ``"auto"``, ``"strict"``).
    auto_enabled:
        Whether automatic mode progression is enabled.
    """
    preset = _detect_preset(rollout_mode, auto_enabled)
    return SafetyTuningSnapshot(
        preset=preset,
        rollout_mode=rollout_mode,  # type: ignore[arg-type]
        auto_enabled=auto_enabled,
    )


def resolve_safety_tuning_from_config(config: dict[str, Any]) -> SafetyTuningSnapshot:
    """Build a tuning snapshot from a config dict with a ``safety`` key.

    Expects ``config["safety"]["rolloutMode"]`` and
    ``config["safety"]["auto"]["enabled"]``.
    """
    safety = config.get("safety", {})
    rollout_mode = safety.get("rolloutMode", "auto")
    auto_enabled = bool(safety.get("auto", {}).get("enabled", True))
    return resolve_safety_tuning(rollout_mode=rollout_mode, auto_enabled=auto_enabled)


def validate_safety_tuning_patch(preset_str: str) -> SafetyTuningPatch:
    """Validate and return a ``SafetyTuningPatch``.

    Raises ``ValueError`` if the preset is not recognised.
    """
    preset = _parse_preset(preset_str)
    if preset is None:
        raise ValueError(f"Invalid safety tuning preset: {preset_str!r}")
    return SafetyTuningPatch(preset=preset)


def apply_safety_tuning_patch(
    config: dict[str, Any],
    patch: SafetyTuningPatch,
) -> dict[str, Any]:
    """Apply a tuning patch to a config dict, returning a new dict.

    Updates ``config.safety.rolloutMode`` and ``config.safety.auto.enabled``
    based on the preset values.
    """
    values = SAFETY_TUNING_PRESETS[patch.preset]
    safety = dict(config.get("safety", {}))
    safety["rolloutMode"] = values.rollout_mode
    auto = dict(safety.get("auto", {}))
    auto["enabled"] = values.auto_enabled
    safety["auto"] = auto

    return {**config, "safety": safety}
