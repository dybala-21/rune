"""Load environment files with process > project > user precedence."""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import overload

from rune.utils.paths import rune_home

# Typed getters (shared across agent/bench/capabilities modules)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str) -> bool:
    """Return True when ``name`` is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


@overload
def env_int(name: str, default: int) -> int: ...
@overload
def env_int(name: str, default: None = ...) -> int | None: ...
def env_int(name: str, default: int | None = None) -> int | None:
    """Read a positive integer, falling back to default when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def env_float(name: str, default: float) -> float:
    """Return ``name`` as a float, else ``default`` (unset or non-numeric)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# Paths

PROJECT_ENV_PATH = Path.cwd() / ".rune" / ".env"


def _user_env_path() -> Path:
    return rune_home() / ".env"


def _project_env_path() -> Path:
    return Path.cwd() / ".rune" / ".env"


USER_ENV_PATH = _user_env_path()

_VALID_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Serialize file edits so concurrent writes cannot discard each other's changes.
_env_write_lock = threading.Lock()

_SENSITIVE_DIR_MODE = 0o700
_SENSITIVE_FILE_MODE = 0o600

EnvConfig = dict[str, str]


# Permission Hardening


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
    """Replace an env file atomically through a sibling created with mode 0600."""
    _ensure_sensitive_dir(file_path.parent)

    fd, tmp_name = tempfile.mkstemp(dir=str(file_path.parent), prefix=".env-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        _harden_permissions(Path(tmp_name), _SENSITIVE_FILE_MODE)
        os.replace(tmp_name, file_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# Parser


def _split_env_line(line: str) -> tuple[str, str] | None:
    """Parse an optional export assignment, respecting quotes and inline comments."""
    trimmed = line.strip()
    if not trimmed or trimmed.startswith("#"):
        return None

    if trimmed.startswith("export "):
        trimmed = trimmed[len("export "):].lstrip()

    key, sep, value = trimmed.partition("=")
    if not sep:
        return None

    key = key.strip()
    if not key or not _VALID_KEY.match(key):
        return None

    value = value.strip()
    if value[:1] == '"':
        # Find the closing quote with serializer-compatible escaping, then decode the value.
        escapes = {"n": "\n", "r": "\r", "t": "\t"}
        out: list[str] = []
        i = 1
        while i < len(value):
            ch = value[i]
            if ch == "\\" and i + 1 < len(value):
                nxt = value[i + 1]
                out.append(escapes.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                break
            out.append(ch)
            i += 1
        # An unterminated quote just takes the rest of the line.
        return key, "".join(out)
    if value[:1] == "'":
        # Single quotes are literal, as in the shell: no escapes inside.
        end = value.find("'", 1)
        if end > 0:
            return key, value[1:end]
        value = value[1:]
    else:
        comment = re.search(r"\s#", value)
        if comment:
            value = value[: comment.start()].rstrip()

    return key, value


def _parse_env_file(content: str) -> EnvConfig:
    """Parse a .env file content string into a dict."""
    env: EnvConfig = {}
    for line in content.split("\n"):
        pair = _split_env_line(line)
        if pair is not None:
            env[pair[0]] = pair[1]
    return env


def _read_for_edit(file_path: Path) -> str:
    """Read an env file for editing; fail rather than overwrite unreadable content."""
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def _read_env_file(file_path: Path) -> EnvConfig:
    """Read and parse a .env file. Returns empty dict on failure."""
    from rune.cloud.boundary import hosted
    if hosted() and file_path == _project_env_path():
        return {}
    try:
        content = file_path.read_text(encoding="utf-8")
        return _parse_env_file(content)
    except (OSError, UnicodeDecodeError):
        return {}


# Loader


def load_env() -> EnvConfig:
    """Merge env files and fill unset process variables; project values override user values."""
    user_env = _read_env_file(_user_env_path())
    project_env = _read_env_file(_project_env_path())

    merged = {**user_env, **project_env}

    for key, value in merged.items():
        if key not in os.environ:
            os.environ[key] = value

    return merged


# Getter


def get_env(key: str) -> str | None:
    """Look up a variable in process, project and user environments, in that order."""
    val = os.environ.get(key)
    if val:
        return val

    project_env = _read_env_file(_project_env_path())
    if key in project_env:
        return project_env[key]

    user_env = _read_env_file(_user_env_path())
    return user_env.get(key)


# Writer


def _effective_scope(key: str, user: EnvConfig, project: EnvConfig) -> str | None:
    scope = "project" if key in project else "user" if key in user else None
    stored = (project if scope == "project" else user).get(key)
    if key in os.environ and os.environ[key] != stored:
        return "process"
    return scope


def effective_env_scope(key: str) -> str | None:
    """Identify the active value's scope without exposing its contents."""
    return _effective_scope(key, _read_env_file(_user_env_path()), _read_env_file(_project_env_path()))


def set_env(key: str, value: str, scope: str = "user") -> None:
    """Set a file and process variable; effective scope updates the file supplying its value."""
    with _env_write_lock:
        if scope == "effective":
            user = _parse_env_file(_read_for_edit(_user_env_path()))
            project = _parse_env_file(_read_for_edit(_project_env_path()))
            scope = _effective_scope(key, user, project) or "user"
            if scope == "process":
                raise ValueError(f"{key} comes from the launch environment. Update it there and restart Rune.")
        file_path = _user_env_path() if scope == "user" else _project_env_path()
        _ensure_sensitive_dir(file_path.parent)
        current = _read_for_edit(file_path)
        _write_sensitive_env_file(file_path, _apply_env_edits(current, {key: value}))
        os.environ[key] = value


def unset_env(key: str, scope: str = "user") -> None:
    """Remove a variable from the selected user/project file and process environment."""
    file_path = _user_env_path() if scope == "user" else _project_env_path()

    with _env_write_lock:
        existed = file_path.exists()
        content = _apply_env_edits(_read_for_edit(file_path), {key: None})
        if content.strip():
            _write_sensitive_env_file(file_path, content)
        elif existed:
            # The successful read confirmed that no other entries remain.
            file_path.unlink()

    os.environ.pop(key, None)


_NEEDS_QUOTING = ' \t#"\'\n\r'


def _format_env_line(key: str, value: str) -> str:
    """Format one entry, escaping newlines so values cannot inject extra keys."""
    if value and (any(ch in value for ch in _NEEDS_QUOTING) or value != value.strip()):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f'{key}="{escaped}"'
    return f"{key}={value}"


def _serialize_env(env: EnvConfig) -> str:
    """Serialize an env dict into .env file content."""
    if not env:
        return ""
    return "\n".join(_format_env_line(k, v) for k, v in env.items()) + "\n"


def _apply_env_edits(content: str, updates: dict[str, str | None]) -> str:
    """Edit keys without disturbing other lines; None removes all definitions of that key."""
    remaining = dict(updates)
    written: set[str] = set()
    out: list[str] = []

    for line in content.split("\n"):
        pair = _split_env_line(line)
        if pair is None or pair[0] not in remaining:
            out.append(line)
            continue
        key = pair[0]
        if key in written:
            continue  # a duplicate further down would shadow the edit
        written.add(key)
        value = remaining[key]
        if value is not None:
            out.append(_format_env_line(key, value))

    for key in written:
        remaining.pop(key, None)

    # Drop a trailing blank so appended keys do not accumulate empty lines.
    while out and not out[-1].strip():
        out.pop()

    out.extend(
        _format_env_line(k, v) for k, v in remaining.items() if v is not None
    )

    if not any(line.strip() for line in out):
        return ""
    return "\n".join(out) + "\n"


# List


def list_env() -> dict[str, EnvConfig]:
    """Return user, project and merged environment mappings."""
    user = _read_env_file(_user_env_path())
    project = _read_env_file(_project_env_path())
    merged = {**user, **project}
    return {"user": user, "project": project, "merged": merged}


# Paths Export

env_paths = {
    "user": USER_ENV_PATH,
    "project": PROJECT_ENV_PATH,
}


def user_env_path() -> Path:
    """Path to the user-level ``.env`` (``~/.rune/.env``)."""
    return _user_env_path()


def project_env_path() -> Path:
    """Resolve the project .rune/.env against the current working directory on each call."""
    return _project_env_path()


# Masking


def is_secret_like_key(key: str) -> bool:
    """Return True if the key looks like it holds a secret value."""
    upper = key.upper()
    return "KEY" in upper or "SECRET" in upper or "TOKEN" in upper


def mask_value(key: str, value: str) -> str:
    """Mask secret values, retaining four characters at each end only when longer than eight."""
    if not is_secret_like_key(key):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


# Auto-load on import

load_env()
