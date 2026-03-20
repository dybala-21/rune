"""Credential management capability for RUNE.

Ported from src/capabilities/credential.ts - saves API keys and
secrets to env files or the system keyring.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.safety.guardian import get_guardian
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Known credential mappings

KNOWN_CREDENTIALS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "brave": "BRAVE_API_KEY",
    "github": "GITHUB_TOKEN",
    "gitlab": "GITLAB_TOKEN",
    "huggingface": "HF_TOKEN",
    "aws_access_key": "AWS_ACCESS_KEY_ID",
    "aws_secret_key": "AWS_SECRET_ACCESS_KEY",
    "google": "GOOGLE_API_KEY",
    "slack": "SLACK_BOT_TOKEN",
    "discord": "DISCORD_BOT_TOKEN",
    "telegram": "TELEGRAM_BOT_TOKEN",
}


def _resolve_env_key(key: str) -> str:
    """Resolve a friendly credential name to an environment variable name.

    If the key matches a known credential alias (case-insensitive), the
    canonical env var name is returned. Otherwise the key is uppercased
    and returned as-is.
    """
    lower = key.lower().replace("-", "_").replace(" ", "_")
    if lower in KNOWN_CREDENTIALS:
        return KNOWN_CREDENTIALS[lower]
    # Already looks like an env var
    if key == key.upper() and "_" in key:
        return key
    return key.upper()


# Parameter schema

class CredentialSaveParams(BaseModel):
    key: str = Field(description="Credential name or env var key")
    value: str = Field(description="Credential value (masked in logs)")
    scope: str = Field(
        default="user",
        description="Scope: user (global) or project (local .env)",
    )


# Implementation

def _get_env_file_path(scope: str) -> Path:
    """Resolve the .env file path based on scope."""
    if scope == "project":
        return Path.cwd() / ".env"
    # User scope: ~/.rune/.env
    from rune.utils.paths import rune_home
    return rune_home() / ".env"


def _update_env_file(env_path: Path, env_key: str, value: str) -> bool:
    """Add or update a key in a .env file.

    Returns True if the value was changed, False if it was already set.
    """
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: list[str] = []
    if env_path.is_file():
        existing_lines = env_path.read_text().splitlines()

    updated = False
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped.startswith(f"{env_key}="):
            # Replace existing line
            new_lines.append(f"{env_key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{env_key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n")

    # Also set in current process environment
    os.environ[env_key] = value

    return True


async def credential_save(params: CredentialSaveParams) -> CapabilityResult:
    """Save a credential to the env file and current environment."""
    env_key = _resolve_env_key(params.key)

    # Mask the value in logs
    masked = params.value[:4] + "..." if len(params.value) > 4 else "***"
    log.info(
        "credential_save",
        key=params.key,
        env_key=env_key,
        scope=params.scope,
        value=masked,
    )

    # Guardian validation - saving credentials is a sensitive operation
    guardian = get_guardian()
    validation = guardian.validate(f"credential_save:{env_key}")
    if not validation.allowed:
        return CapabilityResult(
            success=False,
            error=f"Blocked by Guardian: {validation.reason}",
        )

    try:
        env_path = _get_env_file_path(params.scope)
        _update_env_file(env_path, env_key, params.value)

        scope_label = "project" if params.scope == "project" else "user"

        return CapabilityResult(
            success=True,
            output=(
                f"Credential saved: {env_key}\n"
                f"Scope: {scope_label}\n"
                f"File: {env_path}\n"
                f"Value: {masked}"
            ),
            metadata={
                "env_key": env_key,
                "scope": params.scope,
                "file": str(env_path),
            },
        )

    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Failed to save credential: {exc}",
        )


# Registration

def register_credential_capabilities(registry: CapabilityRegistry) -> None:
    """Register credential management capabilities."""
    registry.register(CapabilityDefinition(
        name="credential_save",
        description="Save an API key or credential",
        domain=Domain.GENERAL,
        risk_level=RiskLevel.HIGH,
        group="write",
        parameters_model=CredentialSaveParams,
        execute=credential_save,
    ))
