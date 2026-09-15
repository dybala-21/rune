"""Browser sessions and manual controls scoped to each conversation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from rune.agent.run_control import RunControl, control_scope
from rune.capabilities.browser.session import BrowserSession, browser_session
from rune.computer.session import DesktopManager, DesktopSession, desktop_scope
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Computer:
    session_id: str
    browser: BrowserSession = field(default_factory=BrowserSession)
    control: RunControl | None = None
    human: bool = False
    lease: int = 0
    frame_id: str = ""
    frame: bytes = b""
    view: dict = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)
    captured: float = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    journal: Any = None
    instructions: list[str] = field(default_factory=list)
    desktop: DesktopSession | None = None
    access_requested: bool = False
    access_ready: asyncio.Event = field(default_factory=asyncio.Event)

    def status(self) -> dict:
        return {
            "sessionId": self.session_id,
            "runId": self.control.run_id if self.control else None,
            "state": "manual" if self.human else self.control.state if self.control else "idle",
            "lease": self.lease,
            "uncertainAction": self.browser.uncertain_action,
            **self.view,
        }

    def invalidate_frame(self) -> None:
        self.frame_id = ""
        self.frame = b""
        self.view = {}
        self.captured = 0


class Computers:
    def __init__(self, *, idle_seconds: float = 1800, capacity: int = 8,
                 record: Callable[[str, dict], Any] | None = None) -> None:
        self.entries: dict[str, Computer] = {}
        self.idle_seconds = idle_seconds
        self.capacity = capacity
        self.record = record
        self.desktop = DesktopManager()
        self.access_lock = asyncio.Lock()

    async def reap(self) -> None:
        for sid, entry in list(self.entries.items()):
            if entry.desktop and entry.desktop.enabled and time.monotonic() >= entry.desktop.expires:
                if entry.control and entry.desktop.control is entry.control:
                    entry.control.pause()
                await self.desktop.release(entry.desktop)
            if not entry.control and not entry.lock.locked() and time.monotonic() - entry.touched > self.idle_seconds:
                # Remove before awaiting close, so a new run cannot acquire the old page.
                self.entries.pop(sid)
                if entry.desktop:
                    await self.desktop.release(entry.desktop)
                await entry.browser.close()

    async def maintain(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.reap()
            except Exception as exc:
                log.warning("computer_expiry_failed", error=str(exc))

    async def claim(self, session_id: str, run_id: str) -> Computer:
        async with self.access_lock:
            return await self._claim(session_id, run_id)

    async def _claim(self, session_id: str, run_id: str) -> Computer:
        owner = self.desktop.owner
        if owner is not None:
            owner.sync_connection()
        entry = self.entries.get(session_id)
        if entry is None:
            if len(self.entries) >= self.capacity:
                raise RuntimeError("Close an idle Computer session before opening another (limit: 8).")
            entry = Computer(session_id)
            self.entries[session_id] = entry
        async with entry.lock:
            if self.entries.get(session_id) is not entry:
                return await self._claim(session_id, run_id)
            if entry.control or entry.human:
                raise RuntimeError("This conversation already has an active run or manual browser control.")
            if entry.desktop and not entry.desktop.enabled:
                entry.desktop = None
            entry.control = RunControl(run_id)
            entry.instructions.clear()
            entry.lease += 1
            entry.touched = time.monotonic()
            if entry.browser.page is not None:
                entry.browser.needs_observation = True
            return entry

    @asynccontextmanager
    async def bind(self, entry: Computer) -> AsyncIterator[None]:
        assert entry.control is not None
        entry.browser.bound_task = asyncio.current_task()
        try:
            with control_scope(entry.control), desktop_scope(None, entry.control, lambda: self.wait_desktop(entry)):
                async with browser_session(entry.browser):
                    yield
        finally:
            entry.browser.bound_task = None

    async def wait_desktop(self, entry: Computer) -> DesktopSession:
        from rune.computer.protocol import DesktopError
        control = entry.control
        async with self.access_lock:
            if entry.browser.bound_task is not asyncio.current_task():
                raise DesktopError("Desktop access belongs to the active conversation task.")
            if control is None or control.state != "running":
                raise DesktopError("The task is no longer waiting for desktop access.")
            owner = self.desktop.owner
            if owner is not None:
                owner.sync_connection()
            if owner is not None and owner.enabled and time.monotonic() < owner.expires and owner is not entry.desktop:
                raise DesktopError("Another conversation has desktop access. Disconnect its apps before connecting this task.")
            if entry.desktop and entry.desktop.enabled:
                entry.desktop.check(agent=False)
                return entry.desktop
            permissions = await self.desktop.host.request("status")
            if entry.control is not control or control.state != "running":
                raise asyncio.CancelledError
            missing = [name for key, name in (("accessibility", "Accessibility"), ("screenRecording", "Screen Recording"))
                       if not permissions.get(key)]
            if missing:
                guidance = "Open System Settings > Privacy & Security and enable Rune Computer under the required permission. "
                try:
                    opened = await self.desktop.host.request("permissions", guard=control.check)
                    if opened.get("settingsOpened"):
                        guidance = "System Settings was opened for you. Enable Rune Computer under the required permission. "
                except DesktopError as exc:
                    log.warning("desktop_permission_settings_failed", error=str(exc))
                    guidance = f"{exc} {guidance}"
                raise DesktopError(
                    f"Rune Computer needs {' and '.join(missing)} permission. "
                    f"{guidance}Then connect the app in Rune and retry. No app input was sent."
                )
            if entry.control is not control or control.state != "running":
                raise asyncio.CancelledError
            entry.access_ready.clear()
            entry.access_requested = True
        try:
            async with asyncio.timeout(600):
                await entry.access_ready.wait()
                await control.ready.wait()
            if entry.control is not control or control.state == "stopped":
                raise asyncio.CancelledError
            if entry.desktop is None or not entry.desktop.enabled:
                raise DesktopError("Desktop access was not granted. No app input was sent.")
            return entry.desktop
        except TimeoutError as exc:
            raise DesktopError("Desktop access was not connected in time. No app input was sent.") from exc
        finally:
            entry.access_requested = False

    def finish(self, entry: Computer, run_id: str) -> None:
        if entry.control and entry.control.run_id == run_id:
            entry.control.stop()
            entry.control = None
            entry.journal = None
            entry.lease += 1
            entry.human = False
            entry.touched = time.monotonic()
            if entry.browser.page is None and (entry.desktop is None or not entry.desktop.enabled):
                self.entries.pop(entry.session_id, None)

    async def stop(self, run_id: str) -> None:
        for entry in self.entries.values():
            if entry.control and entry.control.run_id == run_id:
                run = entry.control
                run.pause()
                if entry.desktop:
                    entry.desktop.invalidate()
                if entry.access_requested:
                    self.desktop.host.cancel_pending()
                    entry.access_ready.set()
                entry.human = False
                entry.lease += 1
                async with entry.lock:
                    run.stop()
                return

    async def close(self) -> None:
        await self.desktop.close()
        for entry in list(self.entries.values()):
            if entry.control:
                entry.control.stop()
            await entry.browser.close()
        self.entries.clear()

    def get(self, session_id: str) -> Computer:
        entry = self.entries.get(session_id)
        if entry is None:
            raise HTTPException(404, "No browser session is open for this conversation.")
        entry.touched = time.monotonic()
        return entry

    async def snapshot(self, entry: Computer) -> dict:
        from rune.capabilities.browser.helpers import extract_interactive_elements

        browser = entry.browser
        if entry.lock.locked() or browser._operation_lock.locked():
            return entry.status()
        async with entry.lock, browser.operation(), browser_session(browser):
            page = browser.page
            if page is None or page.is_closed():
                entry.invalidate_frame()
                return entry.status()
            if time.monotonic() - entry.captured < 1:
                return entry.status()
            # Preview images remain in memory; they aren't added to model history.
            elements = await extract_interactive_elements(page) if entry.human else []
            frame = await page.screenshot(type="jpeg", quality=70, timeout=4000, scale="css")
            if len(frame) > 5 * 1024 * 1024:
                raise HTTPException(413, "Browser preview exceeds the size limit.")
            entry.frame, entry.frame_id = frame, uuid4().hex
            entry.captured = time.monotonic()
            entry.view = {
                "frameId": entry.frame_id, "url": page.url, "title": await page.title(),
                "capturedAt": int(time.time() * 1000),
                "controls": [{"ref": el.ref, "role": el.role, "name": el.name,
                              "disabled": el.is_disabled} for el in elements if el.name][:100],
            }
            return entry.status()


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessionId: str = Field(min_length=1, max_length=200)
    lease: int = Field(ge=0)
    action: Literal["pause", "takeover", "resume", "close"]
    instruction: str = Field(default="", max_length=8000)
    acknowledgeUnknown: bool = False


class ManualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessionId: str = Field(min_length=1, max_length=200)
    lease: int = Field(ge=0)
    frameId: str = Field(min_length=1, max_length=100)
    action: Literal["click", "type", "select", "check", "uncheck", "scroll"]
    ref: str = Field(default="", max_length=100)
    value: str = Field(default="", max_length=8000)


def computer_router(computers: Computers, auth: object) -> APIRouter:
    router = APIRouter(prefix="/api/computer", dependencies=[Depends(auth)])

    @router.get("/state")
    async def state(session_id: str = Query(alias="sessionId", min_length=1, max_length=200)) -> dict:
        entry = computers.entries.get(session_id)
        if entry is None:
            return {"sessionId": session_id, "state": "unavailable", "lease": 0}
        entry.touched = time.monotonic()
        return await computers.snapshot(entry)

    @router.get("/frame/{frame_id}")
    async def frame(frame_id: str, session_id: str = Query(alias="sessionId", min_length=1, max_length=200)) -> Response:
        entry = computers.get(session_id)
        if frame_id != entry.frame_id or not entry.frame:
            raise HTTPException(404, "The preview has changed. Refresh it.")
        return Response(entry.frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.post("/control")
    async def control(req: ControlRequest) -> dict:
        entry = computers.get(req.sessionId)
        async with entry.lock:
            if req.lease != entry.lease:
                raise HTTPException(409, "Control changed. Refresh the Computer panel.")
            run = entry.control
            if req.action == "pause":
                if not run or run.state not in {"running", "pausing", "paused"}:
                    raise HTTPException(409, "There is no running task to pause.")
                run.pause()
                if entry.desktop:
                    entry.desktop.invalidate()
            elif req.action == "takeover":
                if run and run.state != "paused":
                    raise HTTPException(409, "Pause the task and wait for its current tool to finish.")
                if entry.browser.page is None:
                    raise HTTPException(409, "The browser has not opened yet.")
                entry.human = True
                entry.lease += 1
                async with entry.browser.operation():
                    if entry.browser.elements is not None:
                        await entry.browser.elements.release()
                    entry.browser.needs_observation = True
                entry.invalidate_frame()
            elif req.action == "resume":
                if run and run.state != "paused":
                    raise HTTPException(409, "Wait until the task is paused before resuming.")
                if not run and not entry.human:
                    raise HTTPException(409, "Manual control is not active.")
                if entry.browser.uncertain_action and not req.acknowledgeUnknown:
                    raise HTTPException(409, "Check the previous action's effects and confirm before continuing.")
                if entry.desktop and entry.desktop.uncertain:
                    raise HTTPException(409, "Check the uncertain native action in This Mac before continuing.")
                if entry.desktop:
                    entry.desktop.invalidate()
                if run and computers.record is not None:
                    computers.record("user_steering", {"runId": run.run_id, "instruction": req.instruction,
                                                       "acknowledgeUnknown": req.acknowledgeUnknown})
                async with entry.browser.operation():
                    if entry.browser.elements is not None:
                        await entry.browser.elements.release()
                    entry.browser.needs_observation = entry.browser.page is not None
                    entry.browser.uncertain_action = False
                entry.human = False
                entry.lease += 1
                entry.invalidate_frame()
                if run:
                    if req.instruction.strip():
                        entry.instructions.append(req.instruction.strip())
                    run.resume(req.instruction, browser_changed=entry.browser.page is not None)
                    if entry.desktop:
                        run.pending.append("The native desktop may have changed. Call desktop_observe before proposing any input.")
            else:
                if run:
                    raise HTTPException(409, "Stop the task before closing its browser.")
                await entry.browser.close()
                if entry.desktop:
                    await computers.desktop.release(entry.desktop)
                computers.entries.pop(req.sessionId, None)
                return {"sessionId": req.sessionId, "state": "unavailable", "lease": 0}
            return entry.status()

    @router.post("/action")
    async def action(req: ManualRequest) -> dict:
        from rune.capabilities.browser.capabilities import BrowserActParams, browser_act

        entry = computers.get(req.sessionId)
        async with entry.lock, entry.browser.operation(), browser_session(entry.browser):
            if (not entry.human or entry.lease != req.lease or req.frameId != entry.frame_id
                    or entry.control and entry.control.state != "paused"):
                raise HTTPException(409, "Manual control or the preview changed. Refresh before acting.")
            if req.action != "scroll" and req.ref not in {el["ref"] for el in entry.view.get("controls", [])}:
                raise HTTPException(409, "Choose a control from the current page.")
            entry.invalidate_frame()  # Consumed even if the action's acknowledgement is lost.
            entry.browser.needs_observation = False
            try:
                params = BrowserActParams(action=req.action, selector=req.ref, value=req.value)
                if entry.journal is not None:
                    result = await entry.journal.execute("browser_act", {**params.model_dump(), "actor": "user"},
                                                         lambda: browser_act(params))
                else:
                    result = await browser_act(params)
            finally:
                entry.browser.needs_observation = True
            if not result.success:
                raise HTTPException(409, result.error or "The action did not complete. Inspect the page.")
            return entry.status()

    return router
