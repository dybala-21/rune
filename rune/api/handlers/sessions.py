"""Sessions handler - GET /sessions, POST /sessions, GET /sessions/{id}, DELETE /sessions/{id}.

Ported from src/api/handlers/sessions.ts - session (conversation) CRUD
backed by the MemoryStore (SQLite).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.memory.store import get_memory_store
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])
auth = TokenAuthDependency()


# Models


class SessionInfoResponse(BaseModel):
    id: str
    user_id: str = Field("local", alias="userId")
    title: str = ""
    status: str = "active"
    channel: str = "api"
    turn_count: int = Field(0, alias="turnCount")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class SessionListResponse(BaseModel):
    sessions: list[SessionInfoResponse]
    total: int


class SessionCreateRequest(BaseModel):
    title: str = ""
    channel: str = "api"


class SessionCreateResponse(BaseModel):
    id: str
    created: bool


class TurnInfo(BaseModel):
    role: str
    content: str
    channel: str = "api"
    timestamp: str


class SessionDetailResponse(SessionInfoResponse):
    turns: list[TurnInfo] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    id: str
    deleted: bool


# Routes


@router.get("", response_model=SessionListResponse, dependencies=[Depends(auth)])
async def list_sessions(
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> SessionListResponse:
    """List sessions with optional status filter and pagination."""
    store = get_memory_store()

    total = store.count_conversations(status=status)
    rows = store.list_conversations(status=status, limit=limit, offset=offset)

    sessions = [
        SessionInfoResponse(
            id=r["id"],
            userId=r.get("user_id") or "local",
            title=r.get("title") or "",
            status=r.get("status") or "active",
            channel="api",
            turnCount=store.count_conversation_turns(r["id"]),
            createdAt=r.get("created_at") or "",
            updatedAt=r.get("updated_at") or r.get("created_at") or "",
        )
        for r in rows
    ]

    return SessionListResponse(sessions=sessions, total=total)


@router.post("", response_model=SessionCreateResponse, dependencies=[Depends(auth)])
async def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    """Create a new session."""
    store = get_memory_store()
    session_id = store.create_conversation(user_id="local", title=req.title)

    log.info("session_created", session_id=session_id)
    return SessionCreateResponse(id=session_id, created=True)


@router.get("/{session_id}", response_model=SessionDetailResponse, dependencies=[Depends(auth)])
async def get_session(
    session_id: str,
    include_turns: bool = False,
    max_turns: int = 50,
) -> SessionDetailResponse:
    """Get a session by ID, optionally including turn history."""
    store = get_memory_store()
    conv = store.get_conversation(session_id)

    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")

    turns: list[TurnInfo] = []
    turn_count = store.count_conversation_turns(session_id)

    if include_turns:
        raw_turns = store.get_conversation_turns(session_id, limit=max_turns)
        turns = [
            TurnInfo(
                role=t["role"],
                content=t["content"],
                channel=t.get("channel") or "api",
                timestamp=t.get("timestamp") or "",
            )
            for t in raw_turns
        ]

    return SessionDetailResponse(
        id=conv["id"],
        userId=conv.get("user_id") or "local",
        title=conv.get("title") or "",
        status=conv.get("status") or "active",
        channel="api",
        turnCount=turn_count,
        createdAt=conv.get("created_at") or "",
        updatedAt=conv.get("updated_at") or conv.get("created_at") or "",
        turns=turns,
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse, dependencies=[Depends(auth)])
async def delete_session(session_id: str) -> SessionDeleteResponse:
    """Delete a session by ID."""
    store = get_memory_store()
    deleted = store.delete_conversation(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    log.info("session_deleted", session_id=session_id)
    return SessionDeleteResponse(id=session_id, deleted=True)
