"""Pause tool dispatch and discard plans made before a user intervention."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

_current: ContextVar[RunControl | None] = ContextVar("run_control", default=None)
_revision: ContextVar[int] = ContextVar("plan_revision", default=0)


class ControlChanged(RuntimeError):
    pass


class RunControl:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.revision = 0
        self.state = "running"
        self.active = 0
        self.ready = asyncio.Event()
        self.ready.set()
        self.drained = asyncio.Event()
        self.drained.set()
        self.pending: list[str] = []

    def check(self) -> None:
        if self.state != "running" or _revision.get() != self.revision:
            raise ControlChanged("Not executed: the user changed execution control. Read the updated instructions before acting.")

    @asynccontextmanager
    async def dispatch(self) -> AsyncIterator[None]:
        self.check()
        self.active += 1
        self.drained.clear()
        try:
            yield
        finally:
            self.active -= 1
            if not self.active:
                self.drained.set()
                if self.state == "pausing":
                    self.state = "paused"

    def pause(self) -> None:
        if self.state == "running":
            self.revision += 1
            self.ready.clear()
            self.state = "pausing" if self.active else "paused"

    def resume(self, instruction: str = "", *, browser_changed: bool = False) -> None:
        if self.state != "paused":
            raise ControlChanged("Wait for the current tool to finish before resuming.")
        notice = ("The user paused execution. Discard earlier plans and follow the updated instructions. "
                  "Previously dispatched actions may already have taken effect; do not repeat them without checking.")
        if browser_changed:
            notice += " The browser may have changed. Read the current page with browser_observe before changing it."
        self.pending.append(notice)
        if instruction.strip():
            self.pending.append(instruction.strip())
        self.state = "running"
        self.ready.set()

    def stop(self) -> None:
        self.revision += 1
        self.state = "stopped"
        self.ready.set()

    async def checkpoint(self, messages: list[dict[str, Any]]) -> bool:
        await self.ready.wait()
        if self.state == "stopped":
            raise asyncio.CancelledError
        changed = bool(self.pending)
        for content in self.pending:
            messages.append({"role": "user", "content": content})
        self.pending.clear()
        _revision.set(self.revision)
        return changed


def current_control() -> RunControl | None:
    return _current.get()


@contextmanager
def control_scope(control: RunControl) -> Iterator[None]:
    token = _current.set(control)
    revision = _revision.set(control.revision)
    try:
        yield
    finally:
        _revision.reset(revision)
        _current.reset(token)


@asynccontextmanager
async def dispatch_scope() -> AsyncIterator[None]:
    control = current_control()
    if control is None:
        yield
    else:
        async with control.dispatch():
            yield


def controlled_tool(function: Callable) -> Callable:
    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            async with dispatch_scope():
                return await function(*args, **kwargs)
        except ControlChanged as exc:
            return f"ERROR: {exc}"
    return wrapped
