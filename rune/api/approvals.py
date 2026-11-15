"""Approval requests shared by web and streaming clients."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from rune.agent.timing import timed

# The least a sign-in card waits for the owner.
LOGIN_TIMEOUT_MS = 15 * 60 * 1000


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
                       stop: Callable[[], None], *, connect: dict[str, str] | None = None) -> tuple[str, Any]:
        """Show an approval that is decided somewhere else, at `url`, while `done` waits for it.

        With `connect` ({label, detail}) the app shows it as a card to connect an account, not as
        an approval.

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
                    "runId": run_id, "external": {"url": url, **({"connect": connect} if connect else {})},
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

    async def login(command: str, reason: str, login: dict[str, str], kept: Callable[[], Awaitable[Any]],
                    stop: Callable[[], None]) -> tuple[str, Any]:
        """Show a card where the owner signs in to a site in this conversation's browser.

        `login` is {site, url, sessionId, keepUrl}: the card shows the browser of `sessionId` live,
        and, with a `keepUrl`, a link to keep the site on the approval page while `kept` waits for
        that. Returns ("kept", result) when `kept` finishes, ("done", None) when the owner is done
        here without keeping it, ("declined", None) or ("timeout", None). Signing in takes longer
        than an approval, so the card waits at least 15 minutes.
        """
        async with lock:
            approval_id = f"approval:{run_id}:{uuid4().hex}"
            loop = asyncio.get_running_loop()
            waits = max(timeout_ms, LOGIN_TIMEOUT_MS)
            deadline = loop.time() + waits / 1000
            work = asyncio.ensure_future(kept())
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            pending[approval_id] = future
            try:
                await emit("approval_request", {
                    "id": approval_id, "command": command, "riskLevel": "", "reason": reason,
                    "timeoutMs": waits, "expiresAt": time.time() * 1000 + waits, "runId": run_id, "login": login,
                })
                finished, _ = await asyncio.wait({work, future}, timeout=max(0.0, deadline - loop.time()),
                                                 return_when=asyncio.FIRST_COMPLETED)
                if work in finished and work.result() is not None:
                    return "kept", work.result()
                stop()
                if future in finished:
                    return ("done" if approval_granted(future.result()) else "declined"), None
                return "timeout", None
            finally:
                pending.pop(approval_id, None)
                if not future.done():
                    future.cancel()
                if not work.done():
                    work.cancel()
                    with contextlib.suppress(BaseException):
                        await work
                await emit("approval_closed", {"id": approval_id, "runId": run_id})

    request.external = external
    request.login = login
    return request
