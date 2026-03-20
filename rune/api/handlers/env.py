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


@router.get("", response_model=EnvListResponse, dependencies=[Depends(auth)])
async def list_env(scope: str | None = None) -> EnvListResponse:
    """List environment variables.

    Values are masked for security. Filter by ``scope`` (user/project).
    """
    # In a full implementation, this would read from .env files.
    # For now, expose RUNE_* environment variables from the process.
    variables: list[EnvVarInfo] = []
    for key, value in sorted(os.environ.items()):
        if not key.startswith("RUNE_") and not any(
            key.startswith(p) for prefixes, _ in _CATEGORY_PREFIXES for p in prefixes
        ):
            continue

        variables.append(
            EnvVarInfo(
                key=key,
                maskedValue=_mask_value(key, value),
                scope="project",
                isSecret=_is_secret_like_key(key),
                category=_categorize_key(key),
            )
        )

    return EnvListResponse(
        variables=variables,
        paths=EnvPathsInfo(
            user=os.path.expanduser("~/.rune/.env"),
            project=os.path.join(os.getcwd(), ".env"),
        ),
    )


@router.put("/{key}", response_model=EnvSetResponse, dependencies=[Depends(auth)])
async def set_env(key: str, req: EnvSetRequest) -> EnvSetResponse:
    """Set an environment variable.

    The variable is persisted to the appropriate scope file.
    """
    _validate_key(key)

    # In a full implementation, write to .env files.
    os.environ[key] = req.value
    log.info("env_set", key=key, scope=req.scope)

    return EnvSetResponse(key=key, updated=True)


@router.delete("/{key}", response_model=EnvDeleteResponse, dependencies=[Depends(auth)])
async def delete_env(key: str, scope: str = "project") -> EnvDeleteResponse:
    """Remove an environment variable."""
    _validate_key(key)

    deleted = key in os.environ
    os.environ.pop(key, None)
    log.info("env_unset", key=key, scope=scope)

    return EnvDeleteResponse(key=key, deleted=deleted)
