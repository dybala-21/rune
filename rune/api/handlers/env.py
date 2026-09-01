"""Env handler - GET /env, PUT /env/{key}, DELETE /env/{key}.

Ported from src/api/handlers/env.ts - CRUD API for environment
variables. Values are always returned masked for security.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/env", tags=["env"])
auth = TokenAuthDependency()

_VALID_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Category classification by key prefix
_CATEGORY_PREFIXES: list[tuple[list[str], str]] = [
    (["OPENAI_", "ANTHROPIC_", "OLLAMA_"], "llm"),
    (["RUNE_LOG", "RUNE_PRINT"], "logging"),
    (["BRAVE_"], "search"),
    (["TELEGRAM_"], "telegram"),
    (["DISCORD_"], "discord"),
    (["SLACK_"], "slack"),
    (["MATTERMOST_"], "mattermost"),
    (["LINE_"], "line"),
    (["WHATSAPP_"], "whatsapp"),
    (["GOOGLE_CHAT_"], "google-chat"),
]

_SECRET_PATTERNS = {"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL"}


# Helpers


def _categorize_key(key: str) -> str:
    for prefixes, category in _CATEGORY_PREFIXES:
        if any(key.startswith(p) for p in prefixes):
            return category
    return "other"


def _is_secret_like_key(key: str) -> bool:
    upper = key.upper()
    return any(pattern in upper for pattern in _SECRET_PATTERNS)


def _mask_value(key: str, value: str) -> str:
    if _is_secret_like_key(key):
        if len(value) <= 4:
            return "****"
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    return value


def _validate_key(key: str) -> None:
    if not key or not _VALID_KEY_PATTERN.match(key):
        raise HTTPException(
            status_code=400,
            detail=f'Invalid environment variable key: "{key}"',
        )


# Models


class EnvVarInfo(BaseModel):
    key: str
    masked_value: str = Field(alias="maskedValue")
    scope: str
    is_secret: bool = Field(alias="isSecret")
    category: str

    model_config = ConfigDict(populate_by_name=True)


class EnvPathsInfo(BaseModel):
    user: str = ""
    project: str = ""


class EnvListResponse(BaseModel):
    variables: list[EnvVarInfo]
    paths: EnvPathsInfo = Field(default_factory=EnvPathsInfo)


class EnvSetRequest(BaseModel):
    value: str
    scope: Literal["user", "project"] = "project"


class EnvSetResponse(BaseModel):
    key: str
    updated: bool


class EnvDeleteResponse(BaseModel):
    key: str
    deleted: bool


# Routes


def _is_rune_key(key: str) -> bool:
    """Whether a variable belongs to RUNE rather than the ambient shell."""
    if key.startswith("RUNE_"):
        return True
    return any(
        key.startswith(p) for prefixes, _ in _CATEGORY_PREFIXES for p in prefixes
    )


@router.get("", response_model=EnvListResponse, dependencies=[Depends(auth)])
async def list_env(scope: str | None = None) -> EnvListResponse:
    """List environment variables.

    Values are masked. Each variable reports the scope it is actually stored
    in, so the UI can say where an edit will land. Variables only present in
    the process environment report ``process`` — editing one writes a file.
    """
    from rune.utils.env import list_env as read_env_files
    from rune.utils.env import project_env_path, user_env_path

    files = read_env_files()

    # Narrowest scope wins, matching the loader's precedence.
    scopes: dict[str, str] = dict.fromkeys(files["user"], "user")
    scopes.update(dict.fromkeys(files["project"], "project"))

    merged: dict[str, str] = dict(files["merged"])
    for key, value in os.environ.items():
        if _is_rune_key(key):
            merged.setdefault(key, value)
            scopes.setdefault(key, "process")

    variables: list[EnvVarInfo] = []
    for key in sorted(merged):
        if not _is_rune_key(key):
            continue
        var_scope = scopes.get(key, "process")
        if scope and var_scope != scope:
            continue
        variables.append(
            EnvVarInfo(
                key=key,
                maskedValue=_mask_value(key, merged[key]),
                scope=var_scope,
                isSecret=_is_secret_like_key(key),
                category=_categorize_key(key),
            )
        )

    return EnvListResponse(
        variables=variables,
        paths=EnvPathsInfo(user=str(user_env_path()), project=str(project_env_path())),
    )


@router.put("/{key}", response_model=EnvSetResponse, dependencies=[Depends(auth)])
async def set_env(key: str, req: EnvSetRequest) -> EnvSetResponse:
    """Set an environment variable.

    The variable is persisted to the appropriate scope file.
    """
    _validate_key(key)

    # Write the .env file too, not just this process: a variable that vanishes
    # on restart is worse than one that was never accepted.
    from rune.utils.env import set_env as write_env

    try:
        write_env(key, req.value, scope=req.scope)
    except (OSError, UnicodeDecodeError) as exc:
        # The existing file could not be read, so rewriting it would drop
        # whatever else is in there. Refuse rather than destroy it.
        log.warning("env_set_failed", key=key, scope=req.scope, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Could not update the {req.scope} .env file: {exc}",
        ) from exc

    log.info("env_set", key=key, scope=req.scope)
    return EnvSetResponse(key=key, updated=True)


@router.delete("/{key}", response_model=EnvDeleteResponse, dependencies=[Depends(auth)])
async def delete_env(key: str, scope: str = "project") -> EnvDeleteResponse:
    """Remove an environment variable."""
    _validate_key(key)

    from rune.utils.env import list_env as read_env_files
    from rune.utils.env import unset_env

    scope = scope if scope in ("user", "project") else "project"

    files = read_env_files()
    if key not in files[scope]:
        # Report on the file, not the process. A variable inherited from the
        # shell, or set in the other scope, is not ours to remove — saying
        # "deleted" would have it reappear on the next start.
        other = "project" if scope == "user" else "user"
        where = other if key in files[other] else "the environment"
        raise HTTPException(
            status_code=404,
            detail=f'"{key}" is not set in the {scope} .env file (it comes from {where})',
        )

    try:
        unset_env(key, scope=scope)
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("env_unset_failed", key=key, scope=scope, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Could not update the {scope} .env file: {exc}",
        ) from exc

    log.info("env_unset", key=key, scope=scope)
    return EnvDeleteResponse(key=key, deleted=True)
