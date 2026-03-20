"""Tokens handler - POST /tokens, GET /tokens, DELETE /tokens/{id}.

Ported from src/api/handlers/tokens.ts - API token lifecycle management.
Requires admin-level authentication.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import (
    TokenAuthDependency,
    generate_token,
    revoke_token,
)
from rune.api.auth import (
    list_tokens as list_stored_tokens,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/tokens", tags=["tokens"])
auth = TokenAuthDependency()


# Models


class TokenCreateRequest(BaseModel):
    label: str
    permissions: list[str] = Field(default_factory=list)
    expires_seconds: int | None = Field(None, alias="expiresSeconds")

    model_config = ConfigDict(populate_by_name=True)


class TokenCreateResponse(BaseModel):
    token: str
    id: str
    label: str
    permissions: list[str]


class TokenInfoResponse(BaseModel):
    token_prefix: str = Field(alias="tokenPrefix")
    label: str
    created_at: float | None = Field(None, alias="createdAt")
    expires_at: float | None = Field(None, alias="expiresAt")

    model_config = ConfigDict(populate_by_name=True)


class TokenListResponse(BaseModel):
    tokens: list[TokenInfoResponse]


class TokenDeleteResponse(BaseModel):
    id: str
    revoked: bool


# Routes


@router.post("", response_model=TokenCreateResponse, dependencies=[Depends(auth)])
async def create_token(req: TokenCreateRequest) -> TokenCreateResponse:
    """Create a new API token.

    Args (body):
        label: Human-readable label for the token.
        permissions: List of permission scopes.
        expiresSeconds: Optional TTL in seconds.
    """
    if not req.label or not req.label.strip():
        raise HTTPException(status_code=400, detail='Missing or invalid "label"')

    token = generate_token(
        label=req.label.strip(),
        expires_seconds=req.expires_seconds,
    )

    return TokenCreateResponse(
        token=token,
        id=token[:12] + "...",
        label=req.label.strip(),
        permissions=req.permissions,
    )


@router.get("", response_model=TokenListResponse, dependencies=[Depends(auth)])
async def list_tokens() -> TokenListResponse:
    """List all API tokens.

    Token values are truncated for security. Only the prefix is shown.
    """
    stored = list_stored_tokens()
    tokens = [
        TokenInfoResponse(
            tokenPrefix=t.get("token_prefix", ""),
            label=t.get("label", ""),
            createdAt=t.get("created_at"),
            expiresAt=t.get("expires_at"),
        )
        for t in stored
    ]
    return TokenListResponse(tokens=tokens)


@router.delete("/{token_id}", response_model=TokenDeleteResponse, dependencies=[Depends(auth)])
async def delete_token(token_id: str) -> TokenDeleteResponse:
    """Revoke (delete) an API token by its prefix/ID.

    Note: The full token value must be provided since tokens are
    stored by their full value in the current implementation.
    """
    if not token_id:
        raise HTTPException(status_code=400, detail='Missing or invalid "id"')

    success = revoke_token(token_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Token not found: {token_id}")

    log.info("token_revoked_via_api", token_id=token_id[:12])
    return TokenDeleteResponse(id=token_id, revoked=True)
