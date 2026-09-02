"""Configuration loader for RUNE.

Ported from src/config/loader.ts - YAML loading with env var substitution,
mtime caching, deep merge, and fallback to defaults.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rune.config.schema import RuneConfig
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_config: RuneConfig | None = None
_config_mtime: float = 0.0
# (path, mtime) the ignored-key warning was last emitted for.
_warned_ignored_for: tuple[str, float] | None = None


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

    # One parser for both readers. The separate one here read a trailing
    # comment as part of the value, named a variable "export FOO", and turned
    # any prose line containing "=" into one.
    from rune.utils.env import _split_env_line

    for dotenv_path in dotenv_paths:
        if not dotenv_path.is_file():
            continue
        try:
            for line in dotenv_path.read_text().splitlines():
                pair = _split_env_line(line)
                if pair is None:
                    continue
                key, value = pair
                # Don't overwrite existing env vars
                if key not in os.environ:
                    os.environ[key] = value
        except OSError as exc:
            log.debug("dotenv_read_failed", path=str(dotenv_path), error=str(exc))


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

    # Google Gemini API key (simple, like OpenAI)
    if "gemini_api_key" not in data or data["gemini_api_key"] is None:
        data["gemini_api_key"] = os.environ.get("GEMINI_API_KEY")
    if data.get("gemini_api_key"):
        os.environ.setdefault("GEMINI_API_KEY", data["gemini_api_key"])

    # Google Cloud / Vertex AI:
    #   config > env > .rune/google-credentials.json (project)
    #         > ~/.rune/google-credentials.json (user)
    if "google_credentials_file" not in data or data["google_credentials_file"] is None:
        data["google_credentials_file"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if data.get("google_credentials_file") is None:
        project_creds = Path.cwd() / ".rune" / "google-credentials.json"
        if project_creds.is_file():
            data["google_credentials_file"] = str(project_creds)
    if data.get("google_credentials_file") is None:
        default_creds = rune_home() / "google-credentials.json"
        if default_creds.is_file():
            data["google_credentials_file"] = str(default_creds)

    if "vertex_project" not in data or data["vertex_project"] is None:
        data["vertex_project"] = os.environ.get("VERTEX_PROJECT")
    # Auto-detect project from credentials file
    if data.get("vertex_project") is None and data.get("google_credentials_file"):
        try:
            import json as _json
            with open(data["google_credentials_file"]) as f:
                creds = _json.load(f)
            data["vertex_project"] = creds.get("project_id")
        except Exception:
            pass

    if "vertex_location" not in data or data["vertex_location"] is None:
        data["vertex_location"] = os.environ.get("VERTEX_LOCATION", "us-central1")

    # Set env vars so litellm picks them up automatically
    if data.get("google_credentials_file"):
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", data["google_credentials_file"])
    if data.get("vertex_project"):
        os.environ.setdefault("VERTEX_PROJECT", data["vertex_project"])
    if data.get("vertex_location"):
        os.environ.setdefault("VERTEX_LOCATION", data["vertex_location"])

    return data


def _unknown_keys(raw: Any, model: type[BaseModel], path: str = "") -> list[str]:
    """Config-file keys the schema does not declare, as dotted paths.

    Pydantic drops unrecognised keys without a word, so a typo or a setting from
    an older layout looks like it applied and silently does nothing. Surfacing
    them turns that into something the user can see and fix.
    """
    if not isinstance(raw, dict):
        return []

    accepted: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        accepted[name] = field
        if field.alias:
            accepted[field.alias] = field

    unknown: list[str] = []
    for key, value in raw.items():
        if key not in accepted:
            unknown.append(f"{path}{key}")
            continue
        annotation = accepted[key].annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            unknown.extend(_unknown_keys(value, annotation, f"{path}{key}."))
    return unknown


def _validate_salvaging_valid_sections(
    raw: dict[str, Any], cfg_path: Path
) -> RuneConfig:
    """Validate the config, dropping only the sections that fail.

    Pydantic rejects the whole document on one bad value, and the caller's
    fallback then replaced *every* setting with a default — so a typo in one
    block silently cost the user their model, their toggles, everything, with
    only a log line to say so. Retry without the offending top-level sections
    instead, so the rest of the file still applies.
    """
    from pydantic import ValidationError

    try:
        return RuneConfig.model_validate(raw)
    except ValidationError as exc:
        bad_sections = {
            str(err["loc"][0]) for err in exc.errors() if err.get("loc")
        }
        if not bad_sections:
            raise

        salvaged = {k: v for k, v in raw.items() if k not in bad_sections}
        config = RuneConfig.model_validate(salvaged)
        log.warning(
            "config_sections_rejected",
            path=str(cfg_path),
            sections=sorted(bad_sections),
            detail=exc.errors()[0].get("msg", "")[:160],
            hint="these sections were skipped; the rest of the file still applies",
        )
        return config


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

        _config = _validate_salvaging_valid_sections(raw, cfg_path)
        _config_mtime = current_mtime

        log.info("config_loaded", path=str(cfg_path))

        ignored = _unknown_keys(raw, RuneConfig)
        # Once per file version. The list does not change between reloads, and
        # repeating it on every load buries the warnings that do.
        global _warned_ignored_for
        if ignored and _warned_ignored_for != (str(cfg_path), current_mtime):
            _warned_ignored_for = (str(cfg_path), current_mtime)
            log.warning(
                "config_keys_ignored",
                path=str(cfg_path),
                count=len(ignored),
                keys=ignored,
                hint="these keys are not in the schema and had no effect",
            )

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
