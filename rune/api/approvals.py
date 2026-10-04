"""Approval requests shared by web and streaming clients."""

from __future__ import annotations

import asyncio
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

    return request
