"""Browser resources and observations belong to one agent run."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
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


@asynccontextmanager
async def browser_session() -> AsyncIterator[BrowserSession]:
    session = BrowserSession()
    token = _current.set(session)
    try:
        yield session
    finally:
        try:
            await session.close()
        finally:
            _current.reset(token)


def with_browser_session(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        async with browser_session():
            return await function(*args, **kwargs)
    return wrapped


def browser_operation(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        async with current_session().operation():
            return await function(*args, **kwargs)
    return wrapped


async def close_browser_sessions() -> None:
    for session in list(_sessions):
        await session.close()
