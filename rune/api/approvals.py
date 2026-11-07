"""Approval requests shared by web and streaming clients."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from rune.agent.timing import timed


def approval_granted(result: dict[str, Any] | None) -> bool:
    return bool(result and str(result.get("decision", "")).strip().lower()
                in {"approve", "approve_once", "approve_always"})


def approval_callback(
    run_id: str,
    pending: dict[str, asyncio.Future[dict[str, Any]]],
    emit: Callable[[str, dict[str, Any]], Awaitable[Any]],
    *, timeout_ms: int,
) -> Callable[[str, str], Awaitable[bool]]:
    lock = asyncio.Lock()

    @timed("approval")
    async def request(command: str, reason: str) -> bool:
        async with lock:
            from rune.safety.approval_request import current_request

            approval_id = f"approval:{run_id}:{uuid4().hex}"
            future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            pending[approval_id] = future
            suspended = False
            try:
                await emit("approval_request", {
                    "id": approval_id, "command": command, "riskLevel": "", "reason": reason,
                    "timeoutMs": timeout_ms, "expiresAt": time.time() * 1000 + timeout_ms,
                    "runId": run_id,
                    **({"action": action} if (action := current_request()) else {}),
                })
                return approval_granted(await asyncio.wait_for(future, timeout=timeout_ms / 1000))
            except TimeoutError:
                return False
            except asyncio.CancelledError:
                suspended = True
                raise
            finally:
                pending.pop(approval_id, None)
                if not future.done():
                    future.cancel()
                await emit("approval_closed", {"id": approval_id, "runId": run_id,
                                               **({"suspended": True} if suspended else {})})

    async def external(command: str, reason: str, url: str, done: Callable[[], Awaitable[Any]],
                       stop: Callable[[], None]) -> tuple[str, Any]:
        """Show an approval that is decided somewhere else, at `url`, while `done` waits for it.

        Returns ("done", result) when `done` finishes, ("declined", None) when the user declines
        here, or ("timeout", None). Approving here does nothing: that is the point of an approval
        page this process can't reach. On a decline or timeout `stop` asks `done` to finish, and
        `done` returns None unless something it had already sent ran anyway.
        """
        async with lock:
            approval_id = f"approval:{run_id}:{uuid4().hex}"
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_ms / 1000
            work = asyncio.ensure_future(done())
            try:
                await emit("approval_request", {
                    "id": approval_id, "command": command, "riskLevel": "", "reason": reason,
                    "timeoutMs": timeout_ms, "expiresAt": time.time() * 1000 + timeout_ms,
                    "runId": run_id, "external": {"url": url},
                })
                while True:
                    future: asyncio.Future[dict[str, Any]] = loop.create_future()
                    pending[approval_id] = future
                    finished, _ = await asyncio.wait({work, future}, timeout=max(0.0, deadline - loop.time()),
                                                     return_when=asyncio.FIRST_COMPLETED)
                    if work in finished:
                        return "done", work.result()
                    if future in finished and approval_granted(future.result()):
                        continue
                    outcome = "declined" if future in finished else "timeout"
                    stop()
                    # A copy already sent may have run after the owner approved; its answer, not the
                    # decline, says what happened.
                    result = await work
                    return ("done", result) if result is not None else (outcome, None)
            finally:
                pending.pop(approval_id, None)
                if not work.done():
                    work.cancel()
                    with contextlib.suppress(BaseException):
                        await work
                await emit("approval_closed", {"id": approval_id, "runId": run_id})

    request.external = external
    return request
