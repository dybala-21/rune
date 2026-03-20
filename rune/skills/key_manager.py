"""Signing key management for RUNE skills.

Resolves the skill-signing secret from environment variables, key files,
or by generating a new random key.  Follows a precedence order:

1. ``RUNE_SKILL_SIGNING_KEY`` environment variable.
2. Key file on disk (default ``~/.config/rune/keys/skill-signing.key``).
3. Auto-generate and persist a new key (if ``create_if_missing`` is set).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_SKILL_SIGNING_ENV = "RUNE_SKILL_SIGNING_KEY"
SKILL_SIGNING_KEY_FILE_ENV = "RUNE_SKILL_SIGNING_KEY_FILE"
_DEFAULT_KEY_FILENAME = "skill-signing.key"


@dataclass(slots=True)
class SigningSecretResolution:
    """Outcome of resolving the signing secret."""

    secret: str | None = None
    source: Literal["env", "file", "generated", "unavailable"] = "unavailable"
    key_path: str | None = None


def _default_config_dir() -> str:
    """Return the platform-appropriate RUNE config directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return os.path.join(xdg, "rune")
    return os.path.join(Path.home(), ".config", "rune")


def _resolve_key_file_path() -> str:
    override = os.environ.get(SKILL_SIGNING_KEY_FILE_ENV, "").strip()
    if override:
        return override
    return os.path.join(_default_config_dir(), "keys", _DEFAULT_KEY_FILENAME)


def _read_secret_from_file(key_path: str) -> str | None:
    try:
        content = Path(key_path).read_text(encoding="utf-8").strip()
        return content if content else None
    except OSError:
        return None


def _write_generated_secret(key_path: str, secret: str) -> bool:
    try:
        path = Path(key_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use exclusive creation to avoid race conditions
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, f"{secret}\n".encode())
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return True
    except OSError:
        return False


def resolve_skill_signing_secret(
    *,
    env_var_name: str = DEFAULT_SKILL_SIGNING_ENV,
    create_if_missing: bool = True,
) -> SigningSecretResolution:
    """Resolve the skill-signing secret.

    Parameters:
        env_var_name: Name of the environment variable to check first.
        create_if_missing: If ``True``, generate and persist a new key
            when no existing secret is found.

    Returns:
        A :class:`SigningSecretResolution` describing the outcome.
    """
    # 1. Check environment variable
    env_value = os.environ.get(env_var_name, "").strip()
    if env_value:
        return SigningSecretResolution(secret=env_value, source="env")

    # 2. Check key file
    key_path = _resolve_key_file_path()
    from_file = _read_secret_from_file(key_path)
    if from_file:
        return SigningSecretResolution(
            secret=from_file, source="file", key_path=key_path,
        )

    # 3. Generate if allowed
    if not create_if_missing:
        return SigningSecretResolution(source="unavailable", key_path=key_path)

    generated = secrets.token_hex(32)
    written = _write_generated_secret(key_path, generated)
    if not written:
        return SigningSecretResolution(source="unavailable", key_path=key_path)

    # Re-read to account for potential race (another process wrote first)
    materialized = _read_secret_from_file(key_path)
    if not materialized:
        return SigningSecretResolution(source="unavailable", key_path=key_path)

    return SigningSecretResolution(
        secret=materialized, source="generated", key_path=key_path,
    )
