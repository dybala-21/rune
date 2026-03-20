"""Configuration loader for RUNE.

Ported from src/config/loader.ts - YAML loading with env var substitution,
mtime caching, deep merge, and fallback to defaults.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from rune.config.schema import RuneConfig
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_config: RuneConfig | None = None
_config_mtime: float = 0.0


def _env_substitute(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    def _replacer(m: re.Match[str]) -> str:
        var_name = m.group(1)
        return os.environ.get(var_name, m.group(0))
    return re.sub(r"\$\{(\w+)}", _replacer, value)


def _deep_substitute(obj: Any) -> Any:
    """Recursively apply env var substitution to all string values."""
    if isinstance(obj, str):
        return _env_substitute(obj)
    if isinstance(obj, dict):
        return {k: _deep_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_substitute(v) for v in obj]
    return obj


def _load_dotenv() -> None:
    """Load .env files from ~/.rune/.env and .rune/.env (project-level).

    Mimics the TS loadEnv() - existing env vars take priority.
    """
    dotenv_paths = [
        rune_home() / ".env",          # user-level
        Path.cwd() / ".rune" / ".env", # project-level (higher priority)
    ]

    for dotenv_path in dotenv_paths:
        if not dotenv_path.is_file():
            continue
        try:
            for line in dotenv_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                # Don't overwrite existing env vars
                if key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass


def _find_config_file() -> Path | None:
    """Locate the RUNE config file (project-level then user-level)."""
    # Project-level: .rune/config.yaml
    for name in ("config.yaml", "config.yml"):
        project_cfg = Path.cwd() / ".rune" / name
        if project_cfg.is_file():
            return project_cfg

    # User-level: ~/.rune/config.yaml
    for name in ("config.yaml", "config.yml"):
        user_cfg = rune_home() / name
        if user_cfg.is_file():
            return user_cfg

    return None


def _resolve_api_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve API keys from config or environment variables."""
    if "openai_api_key" not in data or data["openai_api_key"] is None:
        data["openai_api_key"] = os.environ.get("OPENAI_API_KEY")

    if "anthropic_api_key" not in data or data["anthropic_api_key"] is None:
        data["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")

    return data


def load_config(force: bool = False) -> RuneConfig:
    """Load configuration from YAML file with env var substitution.

    Uses mtime caching to avoid re-parsing unchanged files.
    Falls back to defaults on any error.
    """
    global _config, _config_mtime

    # Auto-load .env files (like TS loadEnv())
    _load_dotenv()

    cfg_path = _find_config_file()

    if cfg_path is None:
        if _config is not None and not force:
            return _config
        _config = RuneConfig(**_resolve_api_keys({}))
        return _config

    current_mtime = cfg_path.stat().st_mtime
    if _config is not None and not force and current_mtime == _config_mtime:
        return _config

    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        raw: dict[str, Any] = yaml.load(cfg_path) or {}

        # Environment variable substitution
        raw = _deep_substitute(raw)

        # Resolve API keys
        raw = _resolve_api_keys(raw)

        _config = RuneConfig.model_validate(raw)
        _config_mtime = current_mtime

        log.info("config_loaded", path=str(cfg_path))

    except Exception as exc:
        log.warning("config_load_failed", path=str(cfg_path), error=str(exc))
        if _config is None:
            _config = RuneConfig(**_resolve_api_keys({}))

    return _config


def get_config() -> RuneConfig:
    """Get the current configuration (loads if needed)."""
    if _config is None:
        return load_config()
    return _config


def reset_config() -> None:
    """Reset cached config (for testing)."""
    global _config, _config_mtime
    _config = None
    _config_mtime = 0.0
