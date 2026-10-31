"""List saved tasks in the local conversation's workspace."""

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from rune.agent.task_continuity import task_owner
from rune.api import conversation_wiring
from rune.api.auth import TokenAuthDependency
from rune.memory.task_store import TaskStore
from rune.utils.paths import user_workspace


def task_router(auth: TokenAuthDependency) -> APIRouter:
    router = APIRouter(prefix="/api/tasks", dependencies=[Depends(auth)])

    async def saved_tasks(session_id: str = Query(alias="sessionId", min_length=1, max_length=128),
                          query: str = Query(default="", max_length=2000),
                          offset: int = Query(default=0, ge=0, le=10000)) -> dict[str, Any]:
        manager = conversation_wiring.get_conv_manager()
        if manager is None:
            raise HTTPException(503, "Saved work is unavailable. Try again after reconnecting.")
        conversation = manager._active.get(session_id) or await manager._store.load(session_id)
        if conversation is not None and task_owner(conversation.user_id) != "local":
            raise HTTPException(403, "This conversation is not a local workspace.")
        workspace = str(Path(await conversation_wiring.get_workspace(session_id) or user_workspace()).resolve())
        store = TaskStore()
        read = (partial(store.search, "local", workspace, query, limit=11, offset=offset) if query.strip()
                else partial(store.list, "local", workspace, limit=11, offset=offset, unfinished_first=True))
        rows = await asyncio.to_thread(read)
        return {"workspace": workspace, "hasMore": len(rows) > 10, "tasks": [{
            "id": row["id"], "goal": row["goal"][:800], "status": row["status"],
            "updatedAt": row["updated_at"], "blocker": (row.get("checkpoint") or {}).get("blocker", "")[:800],
            "files": (row.get("checkpoint") or {}).get("files", [])[:12],
        } for row in rows[:10]]}

    router.add_api_route("", saved_tasks, methods=["GET"])
    return router
