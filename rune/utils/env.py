"""Environment Variable Manager.

Ported from src/utils/env.ts -- .env file loading and management.

Priority (highest to lowest):
1. os.environ (system environment variables)
2. .rune/.env (project-level)
3. ~/.rune/.env (user-level global)
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

_HOME = Path.home()
USER_ENV_PATH = _HOME / ".rune" / ".env"
PROJECT_ENV_PATH = Path.cwd() / ".rune" / ".env"

_SENSITIVE_DIR_MODE = 0o700
_SENSITIVE_FILE_MODE = 0o600

EnvConfig = dict[str, str]


# ============================================================================
# Permission Hardening
# ============================================================================


def _harden_permissions(target_path: Path, mode: int) -> None:
    """Best-effort chmod on a path. Non-fatal on failure."""
    with contextlib.suppress(OSError):
        target_path.chmod(mode)


def _ensure_sensitive_dir(dir_path: Path) -> None:
    """Create directory with restricted permissions if it does not exist."""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    _harden_permissions(dir_path, _SENSITIVE_DIR_MODE)


def _write_sensitive_env_file(file_path: Path, content: str) -> None:
    """Write content to an env file with restricted permissions."""
    _ensure_sensitive_dir(file_path.parent)
    file_path.write_text(content, encoding="utf-8")
    _harden_permissions(file_path, _SENSITIVE_FILE_MODE)


# ============================================================================
# Parser
# ============================================================================


def _parse_env_file(content: str) -> EnvConfig:
    """Parse a .env file content string into a dict."""
    env: EnvConfig = {}

    for line in content.split("\n"):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue

        # KEY=VALUE parsing
        eq_idx = trimmed.find("=")
        if eq_idx < 0:
            continue

        key = trimmed[:eq_idx].strip()
        value = trimmed[eq_idx + 1 :].strip()

        # Strip surrounding quotes
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]

        env[key] = value

    return env


def _read_env_file(file_path: Path) -> EnvConfig:
    """Read and parse a .env file. Returns empty dict on failure."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return _parse_env_file(content)
    except (OSError, UnicodeDecodeError):
        return {}


# ============================================================================
# Loader
# ============================================================================


def load_env() -> EnvConfig:
    """Load and merge all .env files.

    Priority: os.environ > project .env > user .env.
    Values are only injected into ``os.environ`` when they are not already set.

    Returns the merged config (project overrides user).
    """
    user_env = _read_env_file(USER_ENV_PATH)
    project_env = _read_env_file(PROJECT_ENV_PATH)

    merged = {**user_env, **project_env}

    for key, value in merged.items():
        if key not in os.environ:
            os.environ[key] = value

    return merged


# ============================================================================
# Getter
# ============================================================================


def get_env(key: str) -> str | None:
    """Get an environment variable by key.

    Checks os.environ first, then project .env, then user .env.
    """
    val = os.environ.get(key)
    if val:
        return val

    project_env = _read_env_file(PROJECT_ENV_PATH)
    if key in project_env:
        return project_env[key]

    user_env = _read_env_file(USER_ENV_PATH)
    return user_env.get(key)


# ============================================================================
# Writer
# ============================================================================


def set_env(key: str, value: str, scope: str = "user") -> None:
    """Set an environment variable in a .env file and os.environ.

    Args:
        key: The variable name.
        value: The variable value.
        scope: ``"user"`` for ``~/.rune/.env`` or ``"project"`` for ``.rune/.env``.
    """
    file_path = USER_ENV_PATH if scope == "user" else PROJECT_ENV_PATH
    _ensure_sensitive_dir(file_path.parent)

    existing = _read_env_file(file_path)
    existing[key] = value

    content = _serialize_env(existing)
    _write_sensitive_env_file(file_path, content)

    os.environ[key] = value


def unset_env(key: str, scope: str = "user") -> None:
    """Remove an environment variable from a .env file and os.environ.

    Args:
        key: The variable name to remove.
        scope: ``"user"`` or ``"project"``.
    """
    file_path = USER_ENV_PATH if scope == "user" else PROJECT_ENV_PATH

    existing = _read_env_file(file_path)
    existing.pop(key, None)

    content = _serialize_env(existing)
    if content.strip():
        _write_sensitive_env_file(file_path, content)
    elif file_path.exists():
        file_path.unlink()

    os.environ.pop(key, None)


def _serialize_env(env: EnvConfig) -> str:
    """Serialize an env dict into .env file content."""
    lines: list[str] = []
    for k, v in env.items():
        if any(ch in v for ch in (" ", "#", "=")):
            lines.append(f'{k}="{v}"')
        else:
            lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n" if lines else ""


# ============================================================================
# List
# ============================================================================


def list_env() -> dict[str, EnvConfig]:
    """List all environment variables from user and project .env files.

    Returns a dict with keys ``"user"``, ``"project"``, and ``"merged"``.
    """
    user = _read_env_file(USER_ENV_PATH)
    project = _read_env_file(PROJECT_ENV_PATH)
    merged = {**user, **project}
    return {"user": user, "project": project, "merged": merged}


# ============================================================================
# Paths Export
# ============================================================================

env_paths = {
    "user": USER_ENV_PATH,
    "project": PROJECT_ENV_PATH,
}


# ============================================================================
# Masking
# ============================================================================


def is_secret_like_key(key: str) -> bool:
    """Return True if the key looks like it holds a secret value."""
    upper = key.upper()
    return "KEY" in upper or "SECRET" in upper or "TOKEN" in upper


def mask_value(key: str, value: str) -> str:
    """Mask a value for safe logging if its key looks secret-like.

    Shows the first 4 and last 4 characters with ``***`` in between for
    values longer than 8 characters. Short values are replaced entirely.
    """
    if not is_secret_like_key(key):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


# ============================================================================
# Auto-load on import
# ============================================================================

load_env()
