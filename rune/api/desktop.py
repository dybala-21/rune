"""Native desktop grants and action review, kept out of the model's tool set."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from rune.computer.protocol import DesktopError


class DesktopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessionId: str = Field(min_length=1, max_length=200)
    action: Literal["grant", "disconnect", "decide", "acknowledge"]
    apps: list[str] = Field(default_factory=list, max_length=12)
    revision: int = Field(default=0, ge=0)
    actionId: str = Field(default="", max_length=100)
    approved: bool = False


def desktop_router(computers, auth, stop_run=None) -> APIRouter:
    router = APIRouter(prefix="/api/desktop", dependencies=[Depends(auth)])

    def entry_for(session_id: str):
        entry = computers.entries.get(session_id)
        if entry is None or entry.desktop is None:
            raise HTTPException(404, "No desktop session for this conversation.")
        return entry

    def same_origin(request: Request) -> None:
        if request.headers.get("origin") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Desktop permissions must be changed from the Rune app.")

    @router.get("/status")
    async def status(session_id: str = Query(alias="sessionId", min_length=1, max_length=200)) -> dict:
        entry = computers.entries.get(session_id)
        session = entry.desktop if entry else None
        state = session.status() if session else {"sessionId": session_id, "enabled": False, "revision": 0}
        return {**state, "accessRequested": bool(entry and entry.access_requested),
                "runId": entry.control.run_id if entry and entry.control else None,
                "runState": entry.control.state if entry and entry.control else "idle"}

    @router.get("/setup")
    async def setup() -> dict:
        from rune.computer.installation import installation_info
        from rune.computer.macos import host_path

        executable = getattr(computers.desktop.host, "executable", None) or host_path()
        installed = await asyncio.to_thread(installation_info, executable)
        try:
            return {"available": True, **installed, **await computers.desktop.host.request("status")}
        except DesktopError as exc:
            return {"available": False, **installed, "error": str(exc), "apps": []}

    def require_idle() -> None:
        if any(entry.control or entry.desktop and entry.desktop.enabled for entry in computers.entries.values()):
            raise HTTPException(409, "Finish tasks and disconnect apps before repairing app access.")

    @router.post("/restart")
    async def restart(request: Request) -> dict:
        same_origin(request)
        async with computers.access_lock:
            require_idle()
            async with computers.desktop.host.lock:
                await computers.desktop.host.close()
            return await setup()

    @router.post("/reveal")
    async def reveal(request: Request) -> dict:
        from rune.computer.macos import host_path

        same_origin(request)
        async with computers.access_lock:
            require_idle()
            executable = getattr(computers.desktop.host, "executable", None) or host_path()
            if not executable.is_file():
                raise HTTPException(409, "Rune Computer is not installed.")
            try:
                process = await asyncio.create_subprocess_exec(
                    "/usr/bin/open", "-R", str(executable.parents[2]),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError as exc:
                raise HTTPException(409, "Could not open Finder to show Rune Computer.") from exc
            try:
                code = await asyncio.wait_for(process.wait(), 5)
            except TimeoutError as exc:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise HTTPException(409, "Finder did not respond. Try again.") from exc
            if code != 0:
                raise HTTPException(409, "Could not show Rune Computer in Finder.")
            return {"revealed": True}

    @router.post("/permissions")
    async def permissions(request: Request) -> dict:
        same_origin(request)
        try:
            return await computers.desktop.host.request("permissions")
        except DesktopError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/frame/{observation}")
    async def frame(observation: str, session_id: str = Query(alias="sessionId", min_length=1, max_length=200)) -> Response:
        session = entry_for(session_id).desktop
        try:
            session.check(agent=False)
        except DesktopError as exc:
            raise HTTPException(404, "Desktop access is no longer active.") from exc
        if not session.enabled or not session.frame or session.view.get("observation") != observation:
            raise HTTPException(404, "The app preview is no longer available.")
        return Response(session.frame, media_type="image/png", headers={"Cache-Control": "no-store"})

    async def apply_control(req: DesktopRequest, request: Request) -> dict:
        same_origin(request)
        try:
            if req.action == "grant":
                from rune.api.computer import Computer
                entry = computers.entries.get(req.sessionId)
                if entry and entry.desktop:
                    entry.desktop.sync_connection()
                waiting = bool(entry and entry.access_requested and entry.control and entry.control.state != "stopped")
                waiting_control = entry.control if waiting else None
                if any(item.control and not (item is entry and waiting) for item in computers.entries.values()):
                    raise HTTPException(409, "Finish current tasks before changing desktop access.")
                if entry and (entry.human or entry.desktop and entry.desktop.enabled):
                    raise HTTPException(409, "Return control or disconnect before changing access.")
                if entry is None and len(computers.entries) >= computers.capacity:
                    raise HTTPException(409, "Close an idle Computer session before opening another.")
                session = await computers.desktop.grant(req.sessionId, req.apps)
                if waiting and (entry.control is not waiting_control or not entry.access_requested or waiting_control.state == "stopped"):
                    await computers.desktop.release(session)
                    raise HTTPException(409, "The waiting task ended before access was granted.")
                entry = entry or Computer(req.sessionId)
                entry.desktop = session
                computers.entries[req.sessionId] = entry
                if waiting:
                    entry.access_ready.set()
            else:
                entry = entry_for(req.sessionId)
                session = entry.desktop
                if req.revision != session.revision:
                    raise HTTPException(409, "Desktop access changed. Refresh the panel.")
                if req.action == "disconnect":
                    active = entry.control is not None and session.control is entry.control
                    run_id = entry.control.run_id if active else None
                    if active:
                        entry.control.pause()
                    await computers.desktop.release(session)
                    if run_id and stop_run is not None:
                        await stop_run(run_id)
                elif req.action == "decide":
                    if entry.control is None or entry.control.state != "running":
                        raise HTTPException(409, "The task is no longer running.")
                    session.decide(req.actionId, req.revision, req.approved)
                else:
                    if entry.control and entry.control.state != "paused":
                        raise HTTPException(409, "Pause the task before checking its previous effects.")
                    if not req.approved or not session.uncertain:
                        raise HTTPException(409, "Confirm that you checked the uncertain action.")
                    async with session.lock:
                        await session.host.request("acknowledge")
                        session.uncertain = False
                        session.invalidate()
            return session.status()
        except DesktopError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/control")
    async def control(req: DesktopRequest, request: Request) -> dict:
        if req.action == "grant":
            async with computers.access_lock:
                return await apply_control(req, request)
        return await apply_control(req, request)

    return router
