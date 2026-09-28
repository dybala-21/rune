"""Browser resources can outlive a run when a conversation owns them."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
from uuid import uuid4
from weakref import WeakSet

from rune.utils.logger import get_logger

log = get_logger(__name__)
_current: ContextVar[BrowserSession | None] = ContextVar("browser_session", default=None)
_sessions: WeakSet[BrowserSession] = WeakSet()


@dataclass(eq=False)
class BrowserSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    browser: Any = None
    page: Any = None
    playwright: Any = None
    profile: str = "managed"
    elements: Any = None
    monitor: Any = None
    observe_history: list[str] = field(default_factory=list)
    profiles: dict[str, dict] = field(default_factory=dict)
    init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _owner: asyncio.Task | None = None
    closed: bool = False
    uncertain_action: bool = False
    needs_observation: bool = False
    bound_task: asyncio.Task | None = None
    native_host: Any = None
    owner_id: str = ""
    native_target: str = ""
    tabs: list[dict] = field(default_factory=list)
    acquire_control: Callable[[], Awaitable[None]] | None = None
    last_input: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _sessions.add(self)

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if self._owner is task:
            yield
            return
        async with self._operation_lock:
            if self.closed:
                raise RuntimeError("This browser session has ended")
            self._owner = task
            try:
                yield
            finally:
                self._owner = None

    async def release_resources(self) -> None:
        if self.elements is not None:
            await self.elements.release()
        for resource, method in ((self.monitor, "detach"), (self.browser, "close"), (self.playwright, "stop")):
            if resource is not None:
                try:
                    await getattr(resource, method)()
                except Exception as exc:
                    log.debug("browser_resource_close_failed", resource=method, error=str(exc))
        self.browser = self.page = self.playwright = self.monitor = None
        if self.profile == "native" and self.native_host:
            try:
                await self.native_host.request(self.owner_id, "close")
            except Exception as exc:
                log.debug("native_browser_close_failed", error=str(exc))
        self.native_target = ""
        self.last_input = None
        self.tabs.clear()
        self.observe_history.clear()

    async def close(self) -> None:
        if self.closed:
            return
        async with self.operation(), self.init_lock:
            self.closed = True
            await self.release_resources()
        _sessions.discard(self)


def current_session() -> BrowserSession:
    session = _current.get()
    if session is None:
        session = BrowserSession()
        _current.set(session)
    if session.closed:
        raise RuntimeError("This browser session has ended")
    return session


async def describe_browser_session() -> dict[str, Any]:
    """Read bounded routing context without opening a browser or taking a screenshot."""
    session = _current.get()
    if session is None or session.page is None or session.closed:
        return {"status": "unavailable"}
    if not session.browser.is_connected():
        return {"status": "closed"}
    try:
        async with asyncio.timeout(1):
            if session.profile == "native":
                from rune.browser.native import select_native_page
                await select_native_page(session)
            if session.page.is_closed():
                return {"status": "closed"}
            title = await session.page.title()
            return {"status": "open", "url": session.page.url[:1000], "title": title[:160],
                    "needs_observation": session.needs_observation}
    except Exception as exc:
        log.debug("browser_context_unavailable", error=type(exc).__name__)
        return {"status": "unknown"}


@asynccontextmanager
async def browser_session(session: BrowserSession | None = None) -> AsyncIterator[BrowserSession]:
    owned = session is None
    session = session or BrowserSession()
    token = _current.set(session)
    try:
        yield session
    finally:
        try:
            if owned:
                await session.close()
        finally:
            _current.reset(token)


def with_browser_session(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        session = _current.get()
        if session is not None and session.bound_task is asyncio.current_task():
            return await function(*args, **kwargs)
        async with browser_session():
            return await function(*args, **kwargs)
    return wrapped


def browser_operation(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        from rune.agent.run_control import current_control
        session = current_session()
        control = current_control()
        if control is not None and session.acquire_control and session._owner is not asyncio.current_task():
            await session.acquire_control()
        async with session.operation():
            if control is not None:
                control.check()
            return await function(*args, **kwargs)
    return wrapped


async def close_browser_sessions() -> None:
    for session in list(_sessions):
        await session.close()
