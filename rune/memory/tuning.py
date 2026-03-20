"""Memory tuning configuration.

Ported from src/memory/tuning.ts - configurable parameters for semantic
search thresholds, episode limits, and context budgets. Supports env-var
overrides and named presets.
"""

from __future__ import annotations

import os
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Defaults

MEMORY_TUNING_DEFAULTS: dict[str, int | float] = {
    "semantic_limit": 5,
    "semantic_min_score": 0.3,
    "uncertain_semantic_limit": 8,
    "uncertain_semantic_min_score": 0.2,
    "max_episodes": 10,
    "context_max_chars": 8000,
}

# Environment variable mapping

MEMORY_TUNING_ENV_KEYS: dict[str, str] = {
    "semantic_limit": "RUNE_MEMORY_SEMANTIC_LIMIT",
    "semantic_min_score": "RUNE_MEMORY_SEMANTIC_MIN_SCORE",
    "uncertain_semantic_limit": "RUNE_MEMORY_UNCERTAIN_SEMANTIC_LIMIT",
    "uncertain_semantic_min_score": "RUNE_MEMORY_UNCERTAIN_SEMANTIC_MIN_SCORE",
    "max_episodes": "RUNE_MEMORY_MAX_EPISODES",
    "context_max_chars": "RUNE_MEMORY_CONTEXT_MAX_CHARS",
}

# Presets

MemoryTuningPreset = Literal["minimal", "balanced", "aggressive", "research"]

PRESETS: dict[str, dict[str, int | float]] = {
    "minimal": {
        "semantic_limit": 2,
        "semantic_min_score": 0.5,
        "uncertain_semantic_limit": 3,
        "uncertain_semantic_min_score": 0.4,
        "max_episodes": 3,
        "context_max_chars": 3000,
    },
    "balanced": {
        # Same as defaults - explicitly stated for clarity
        "semantic_limit": 5,
        "semantic_min_score": 0.3,
        "uncertain_semantic_limit": 8,
        "uncertain_semantic_min_score": 0.2,
        "max_episodes": 10,
        "context_max_chars": 8000,
    },
    "aggressive": {
        "semantic_limit": 10,
        "semantic_min_score": 0.15,
        "uncertain_semantic_limit": 15,
        "uncertain_semantic_min_score": 0.1,
        "max_episodes": 20,
        "context_max_chars": 16000,
    },
    "research": {
        "semantic_limit": 15,
        "semantic_min_score": 0.1,
        "uncertain_semantic_limit": 20,
        "uncertain_semantic_min_score": 0.05,
        "max_episodes": 50,
        "context_max_chars": 32000,
    },
}


# Env-var parsers

def parse_probability_env(name: str, fallback: float) -> float:
    """Parse an env var as a float in [0.0, 1.0], returning *fallback* on error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        val = float(raw)
        if not (0.0 <= val <= 1.0):
            log.warning("env_probability_out_of_range", name=name, value=val)
            return fallback
        return val
    except ValueError:
        log.warning("env_probability_parse_failed", name=name, raw=raw)
        return fallback


def parse_positive_int_env(name: str, fallback: int) -> int:
    """Parse an env var as a positive integer, returning *fallback* on error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        val = int(raw)
        if val <= 0:
            log.warning("env_positive_int_out_of_range", name=name, value=val)
            return fallback
        return val
    except ValueError:
        log.warning("env_positive_int_parse_failed", name=name, raw=raw)
        return fallback


# Config builder

_FLOAT_KEYS = frozenset({
    "semantic_min_score",
    "uncertain_semantic_min_score",
})

_INT_KEYS = frozenset({
    "semantic_limit",
    "uncertain_semantic_limit",
    "max_episodes",
    "context_max_chars",
})


def get_tuning_config(preset: MemoryTuningPreset | None = None) -> dict[str, int | float]:
    """Build a tuning config by merging defaults + preset + env overrides.

    Precedence (highest wins): environment variables > preset > defaults.
    """
    # Start with defaults
    config: dict[str, int | float] = dict(MEMORY_TUNING_DEFAULTS)

    # Layer preset overrides
    if preset is not None:
        preset_values = PRESETS.get(preset)
        if preset_values is None:
            log.warning("unknown_tuning_preset", preset=preset)
        else:
            config.update(preset_values)

    # Layer environment variable overrides
    for key, env_name in MEMORY_TUNING_ENV_KEYS.items():
        if key in _FLOAT_KEYS:
            default_val = config.get(key, 0.0)
            env_val = parse_probability_env(env_name, float(default_val))
            if env_val != float(default_val) or os.environ.get(env_name, "").strip():
                config[key] = env_val
        elif key in _INT_KEYS:
            default_val = config.get(key, 0)
            env_val = parse_positive_int_env(env_name, int(default_val))
            if env_val != int(default_val) or os.environ.get(env_name, "").strip():
                config[key] = env_val

    return config
