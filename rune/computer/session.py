"""Conversation grants and one-use approvals for native input."""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from rune.agent.run_control import ControlChanged, RunControl, current_control
from rune.agent.tool_output import ToolImage
from rune.computer.macos import MacHost
from rune.computer.observation import ObservationProgress, match_condition
from rune.computer.protocol import DesktopAction, DesktopCondition, DesktopError, DesktopWait

TOOLS = frozenset({"desktop_apps", "desktop_open", "desktop_observe", "desktop_wait", "desktop_act", "think", "ask_user"})
_current: ContextVar[DesktopSession | None] = ContextVar("desktop_session", default=None)
_access: ContextVar[Callable[[], Awaitable[DesktopSession]] | None] = ContextVar("desktop_access", default=None)


def current_desktop() -> DesktopSession | None:
    return _current.get()


async def require_desktop() -> None:
    if current_desktop() is not None:
        current_desktop().check()
        return
    request = _access.get()
    if request is None:
        raise DesktopError("Native app work requires the Rune Computer panel. Connect the app there before continuing.")
    session = await request()
    session.check(agent=False)
    session.begin_task()
    session.task = asyncio.current_task()
    session.control = current_control()
    session.invalidate()
    _current.set(session)


@dataclass
class DesktopSession:
    session_id: str
    host: Any
    apps: dict[str, str]
    expires: float
    revision: int = 1
    enabled: bool = True
    uncertain: bool = False
    native_review: bool = False
    waiting: bool = False
    view: dict = field(default_factory=dict)
    frame: bytes = b""
    pending: dict | None = None
    task: asyncio.Task | None = None
    control: RunControl | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    host_generation: int = field(init=False)
    connection_error: str = ""
    last_error: str = ""
    action_failed: bool = False
    observations: int = 0
    inputs: int = 0
    used: bool = False
    progress: ObservationProgress = field(default_factory=ObservationProgress)
    repeat_block: tuple[str, str, str] | None = None
    unmet_conditions: list[tuple[str, DesktopCondition]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.host_generation = getattr(self.host, "generation", 0)

    def sync_connection(self) -> None:
        if self.enabled and (getattr(self.host, "generation", 0) != self.host_generation
                             or not getattr(self.host, "connected", True)):
            self.connection_error = "Rune Computer disconnected. Reconnect the selected apps before continuing."
            self.revoke()

    def begin_task(self) -> None:
        self.last_error = ""
        self.action_failed = False
        self.observations = 0
        self.inputs = 0
        self.used = False
        self.progress = ObservationProgress()
        self.repeat_block = None
        self.unmet_conditions.clear()

    def completion_blocker(self, *, requires_input: bool = False) -> str | None:
        self.sync_connection()
        if not self.enabled or time.monotonic() >= self.expires:
            return self.connection_error or "Desktop access ended before the task outcome was confirmed."
        if self.uncertain:
            return "A desktop action has an unknown outcome. Inspect the app before continuing."
        if self.last_error or self.action_failed:
            return self.last_error or "The last desktop input was not confirmed."
        if self.unmet_conditions:
            return "A requested screen condition is still unconfirmed. Check that condition in its app before claiming completion."
        if not self.observations or not self.view:
            return "No current app observation confirms the result of this task."
        if requires_input and not self.inputs:
            return "This task required app input, but no desktop input was completed."
        return None

    def check(self, *, agent: bool = True) -> None:
        self.sync_connection()
        if not self.enabled or time.monotonic() >= self.expires:
            raise DesktopError(self.connection_error or "Desktop access expired or was disconnected. Ask the user to reconnect it.")
        if agent:
            if self.task is not asyncio.current_task() or self.control is not current_control():
                raise DesktopError("Desktop access belongs to the active conversation task.")
            if self.control is None:
                raise DesktopError("No active desktop task")
            self.control.check()

    def check_revision(self, revision: int) -> None:
        try:
            self.check()
        except ControlChanged as exc:
            raise DesktopError(str(exc)) from exc
        if revision != self.revision:
            raise DesktopError("Desktop control changed before native dispatch.")

    def invalidate(self, *, clear_view: bool = True) -> None:
        if getattr(self.host, "generation", 0) == self.host_generation:
            self.host.cancel_pending()
        if clear_view:
            self.view = {}
            self.frame = b""
        self.revision += 1
        self.pending = None

    def revoke(self) -> None:
        self.enabled = False
        self.invalidate()

    def status(self) -> dict:
        self.sync_connection()
        return {"sessionId": self.session_id, "enabled": self.enabled and time.monotonic() < self.expires,
                "revision": self.revision, "apps": [{"id": key, "name": value} for key, value in self.apps.items()],
                "uncertainAction": self.uncertain, "nativeReview": self.native_review, "waiting": self.waiting, "pending": self.pending,
                "connectionError": self.connection_error,
                "expiresIn": max(0, int(self.expires - time.monotonic())) if self.enabled else 0, **self.view}

    async def observe(self, app: str, *, open_app: bool = False) -> dict:
        self.used = True
        self.check()
        if app not in self.apps:
            raise DesktopError("The user has not allowed this app.")
        async with self.lock:
            return await self._observe(app, open_app=open_app)

    async def _observe(self, app: str, *, open_app: bool = False) -> dict:
        self.check()
        if self.pending:
            raise DesktopError("Resolve the pending action before observing again.")
        self.invalidate()
        revision = self.revision
        try:
            data = await self.host.request("open" if open_app else "observe", {"app": app}, guard=lambda: self.check_revision(revision))
            self.check_revision(revision)
            if data.get("app") != app:
                raise DesktopError("The native host returned a different app than requested.")
            result = self.accept_view(data)
            if not self.action_failed:
                self.last_error = ""
            return result
        except DesktopError as exc:
            self.last_error = str(exc)
            self.sync_connection()
            raise

    async def wait(self, request: DesktopWait) -> dict:
        self.used = True
        self.check()
        if request.app not in self.apps:
            raise DesktopError("The user has not allowed this app.")
        async with self.lock:
            condition = (request.app, request.condition)
            if condition not in self.unmet_conditions:
                self.unmet_conditions.append(condition)
            loop = asyncio.get_running_loop()
            started = loop.time()
            deadline = started + request.timeout_ms / 1000
            polls = 0
            self.waiting = True
            try:
                while True:
                    self.check()
                    if polls and loop.time() >= deadline:
                        raise TimeoutError
                    # Finish the current reply; cancelling it would desynchronize the native pipe.
                    # MacHost bounds each read separately. The deadline limits new polls.
                    data = await self._observe(request.app)
                    polls += 1
                    evidence = match_condition(request.condition, self.view)
                    if evidence is not None:
                        if (self.repeat_block is not None and self.repeat_block[0] == request.app
                                and self.progress.states[request.app] != self.repeat_block[1]):
                            # A delayed result can resolve a repeat block without another input.
                            self.repeat_block = None
                            self.action_failed = False
                            self.last_error = ""
                        data["conditionCheck"] = {"status": "matched", "scope": "visible_app_state",
                            "condition": request.condition.model_dump(exclude_defaults=True), "evidence": evidence,
                            "polls": polls, "elapsedMs": round((loop.time() - started) * 1000)}
                        self.view["conditionCheck"] = data["conditionCheck"]
                        return data
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError
                    await asyncio.sleep(min(0.25, remaining))
            except TimeoutError as exc:
                self.last_error = "The requested screen condition was not observed within the wait limit. No input was repeated."
                raise DesktopError(self.last_error) from exc
            finally:
                self.waiting = False

    def accept_view(self, data: dict) -> dict:
        if data.get("app") not in self.apps or not data.get("observation"):
            raise DesktopError("The native host returned an unexpected app or observation.")
        try:
            frame = base64.b64decode(data["image_base64"], validate=True)
            ToolImage.from_bytes(frame)
        except (KeyError, ValueError) as exc:
            raise DesktopError("The app screenshot could not be validated.") from exc
        self.frame = frame
        self.observations += 1
        self.unmet_conditions = [(app, condition) for app, condition in self.unmet_conditions
                                 if app != data["app"] or match_condition(condition, data) is None]
        data["progress"] = self.progress.observe(data, frame)
        self.view = {key: value for key, value in data.items() if key != "image_base64"}
        return data

    async def act(self, action: DesktopAction) -> dict:
        self.used = True
        self.check()
        async with self.lock:
            self.check()
            if self.uncertain:
                raise DesktopError("The previous action has an unknown outcome. Ask the user to inspect the app.")
            if not self.frame or action.observation != self.view.get("observation"):
                raise DesktopError("Read the app with desktop_observe before proposing input.")
            input_key = self.progress.action_key(action, self.view)
            blocker = self.progress.blocker(input_key) if action.action != "publish" else None
            if blocker:
                self.last_error, self.action_failed = blocker, True
                self.repeat_block = input_key
                raise DesktopError(blocker)
            self.repeat_block = None
            # Build the approval display from the validated request.
            request = action.model_dump(exclude_defaults=True)
            action_id = uuid4().hex
            self.revision += 1
            revision = self.revision
            self.pending = {"id": action_id, "app": self.view["app"], "appName": self.apps[self.view["app"]],
                            "action": request, "target": next((el for el in self.view.get("controls", []) if el.get("ref") == action.ref), None),
                            "expiresAt": int((time.time() + 120) * 1000)}
            async def watch_review() -> None:
                deadline = time.monotonic() + 120
                while True:
                    await asyncio.sleep(0.1)
                    try:
                        self.check(agent=False)
                        if self.control is not None:
                            self.control.check()
                        if revision != self.revision or time.monotonic() >= deadline:
                            raise DesktopError("The native review expired or control changed.")
                    except (DesktopError, ControlChanged):
                        self.invalidate()
                        return

            dispatched = False
            watcher = asyncio.create_task(watch_review())
            try:
                self.check_revision(revision)
                dispatched = True
                self.native_review = True
                data = await self.host.request("act", request, guard=lambda: self.check_revision(revision))
                try:
                    self.check_revision(revision)
                except DesktopError as exc:
                    raise DesktopError("Control changed while the result was returning. Inspect the app before continuing.",
                                       outcome="not_executed" if action.action == "publish" else "unknown") from exc
                artifact = data.pop("artifact", None)
                try:
                    if data.get("app") != input_key[0]:
                        raise DesktopError("The native host returned a different app after input.")
                    self.accept_view(data)
                except DesktopError as exc:
                    raise DesktopError(str(exc), outcome="not_executed" if action.action == "publish" else "unknown") from exc
                if action.action == "publish":
                    from rune.computer.artifacts import Artifacts
                    try:
                        content = base64.b64decode(artifact["data_base64"], validate=True)
                        data["artifact"] = Artifacts().publish(self.session_id, self.control.run_id,
                            artifact["path"], content, artifact["sha256"])
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        raise DesktopError("The saved document could not be registered for download. Publish it again.") from exc
                self.view = {key: value for key, value in data.items() if key != "image_base64"}
                self.last_error = ""
                self.action_failed = False
                self.repeat_block = None
                if action.action != "publish":
                    self.inputs += 1
                    data["inputObservation"] = self.progress.acted(input_key)
                    self.view["inputObservation"] = data["inputObservation"]
                return data
            except BaseException as exc:
                self.last_error = str(exc)
                self.action_failed = True
                self.sync_connection()
                if dispatched and action.action != "publish" and (not isinstance(exc, DesktopError) or exc.outcome == "unknown"):
                    self.uncertain = True
                raise
            finally:
                self.native_review = False
                self.pending = None
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                if not dispatched:
                    self.invalidate()

    def decide(self, action_id: str, revision: int, approved: bool) -> None:
        self.check(agent=False)
        if revision != self.revision or self.pending is None or self.pending["id"] != action_id:
            raise DesktopError("This approval is no longer current.")
        if approved:
            raise DesktopError("Approve the action in the Rune Computer dialog on your Mac.")
        self.invalidate()


class DesktopManager:
    def __init__(self, host: Any = None) -> None:
        self.host = host or MacHost()
        self.owner: DesktopSession | None = None
        self.lock = asyncio.Lock()

    async def grant(self, session_id: str, apps: list[str]) -> DesktopSession:
        async with self.lock:
            if self.owner is not None:
                self.owner.sync_connection()
            if self.owner is not None and time.monotonic() >= self.owner.expires:
                self.owner.revoke()
                await self.host.close()
                self.owner = None
            if self.owner is not None and self.owner.enabled:
                raise DesktopError("Disconnect the current desktop session before granting new access.")
            catalog = await self.host.request("status")
            if not catalog.get("accessibility") or not catalog.get("screenRecording"):
                raise DesktopError("Enable Accessibility and Screen Recording for Rune Computer in macOS System Settings.")
            available = {app["id"]: app["name"] for app in catalog.get("apps", [])}
            if not apps or any(app not in available for app in apps):
                raise DesktopError("Choose installed apps from the current list.")
            await self.host.request("grant", {"apps": apps})
            self.owner = DesktopSession(session_id, self.host, {app: available[app] for app in apps}, time.monotonic() + 1800)
            return self.owner

    async def release(self, session: DesktopSession) -> None:
        # Fence dispatch before waiting for the host pipe or an in-flight action.
        session.revoke()
        async with self.lock:
            if self.owner is session:
                await self.host.close()
                self.owner = None

    async def close(self) -> None:
        if self.owner is not None:
            await self.release(self.owner)
        else:
            await self.host.close()


@contextmanager
def desktop_scope(session: DesktopSession | None, control: RunControl,
                  request: Callable[[], Awaitable[DesktopSession]] | None = None) -> Iterator[None]:
    token = _current.set(session)
    access_token = _access.set(request)
    if session is not None:
        session.begin_task()
        session.task = asyncio.current_task()
        session.control = control
        session.invalidate()
    try:
        yield
    finally:
        session = _current.get()
        if session is not None:
            session.invalidate(clear_view=False)
            session.task = None
            session.control = None
        _current.reset(token)
        _access.reset(access_token)
