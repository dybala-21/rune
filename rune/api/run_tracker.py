"""Track agent execution runs.

Ported from src/api/run-tracker.ts - in-memory tracking with optional
persistence. Manages RunState lifecycle: queued -> running -> completed/failed/aborted.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rune.api.protocol import RunStatus
from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    cache_read: int | None = None


@dataclass
class RunResult:
    success: bool
    answer: str
    usage: TokenUsage | None = None


@dataclass
class RunState:
    run_id: str
    session_id: str
    client_id: str
    status: RunStatus
    goal: str
    started_at: float  # time.time() epoch
    conversation_id: str | None = None
    completed_at: float | None = None
    result: RunResult | None = None
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        d: dict[str, Any] = {
            "runId": self.run_id,
            "sessionId": self.session_id,
            "clientId": self.client_id,
            "status": self.status,
            "goal": self.goal,
            "startedAt": self.started_at,
        }
        if self.conversation_id:
            d["conversationId"] = self.conversation_id
        if self.completed_at:
            d["completedAt"] = self.completed_at
        if self.result:
            d["result"] = {
                "success": self.result.success,
                "answer": self.result.answer,
            }
            if self.result.usage:
                d["result"]["usage"] = {
                    "input": self.result.usage.input,
                    "output": self.result.usage.output,
                }
                if self.result.usage.cache_read is not None:
                    d["result"]["usage"]["cacheRead"] = self.result.usage.cache_read
        if self.error:
            d["error"] = self.error
        return d


MAX_COMPLETED = 100
COMPLETED_TTL_SEC = 10 * 60  # 10 minutes


class RunTracker:
    """In-memory tracker for agent execution runs.

    Manages active and recently-completed runs. Provides state transition
    methods and query capabilities.
    """

    def __init__(self) -> None:
        self._active: dict[str, RunState] = {}
        self._completed: dict[str, RunState] = {}

    def create(
        self,
        run_id: str,
        client_id: str,
        session_id: str,
        goal: str,
    ) -> RunState:
        """Create a new run in ``queued`` state."""
        state = RunState(
            run_id=run_id,
            session_id=session_id,
            client_id=client_id,
            status="queued",
            goal=goal,
            started_at=time.time(),
        )
        self._active[run_id] = state
        log.info("run_created", run_id=run_id, goal=goal[:80])
        return state

    def mark_running(self, run_id: str) -> None:
        run = self._active.get(run_id)
        if run:
            run.status = "running"
            log.info("run_running", run_id=run_id)

    def mark_completed(self, run_id: str, result: RunResult) -> None:
        run = self._active.get(run_id)
        if not run:
            return
        run.status = "completed"
        run.completed_at = time.time()
        run.result = result
        self._move_to_completed(run)
        log.info("run_completed", run_id=run_id)

    def mark_failed(self, run_id: str, error: str) -> None:
        run = self._active.get(run_id)
        if not run:
            return
        run.status = "failed"
        run.completed_at = time.time()
        run.error = error
        self._move_to_completed(run)
        log.warning("run_failed", run_id=run_id, error=error)

    def mark_aborted(self, run_id: str) -> None:
        run = self._active.get(run_id)
        if not run:
            return
        run.status = "aborted"
        run.completed_at = time.time()
        self._move_to_completed(run)
        log.info("run_aborted", run_id=run_id)

    def abort(self, run_id: str) -> bool:
        """Request cancellation for a run. Returns True if run was found and active."""
        run = self._active.get(run_id)
        if not run:
            return False
        run.cancel_event.set()
        self.mark_aborted(run_id)
        return True

    def link_conversation(self, run_id: str, conversation_id: str) -> None:
        run = self.get(run_id)
        if run:
            run.conversation_id = conversation_id

    def get(self, run_id: str) -> RunState | None:
        return self._active.get(run_id) or self._completed.get(run_id)

    def get_active_count(self) -> int:
        return len(self._active)

    def get_for_client(self, client_id: str) -> list[RunState]:
        result: list[RunState] = []
        for run in self._active.values():
            if run.client_id == client_id:
                result.append(run)
        for run in self._completed.values():
            if run.client_id == client_id:
                result.append(run)
        return result

    def get_all_active(self) -> list[RunState]:
        return list(self._active.values())

    async def query_runs(
        self,
        *,
        client_id: str | None = None,
        conversation_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = 20,
        offset: int | None = 0,
    ) -> list[dict[str, Any]]:
        """Query runs with optional filters."""
        all_runs = list(self._active.values()) + list(self._completed.values())

        if client_id:
            all_runs = [r for r in all_runs if r.client_id == client_id]
        if conversation_id:
            all_runs = [r for r in all_runs if r.conversation_id == conversation_id]
        if status:
            all_runs = [r for r in all_runs if r.status == status]

        # Sort by started_at descending
        all_runs.sort(key=lambda r: r.started_at, reverse=True)

        start = offset or 0
        end = start + (limit or 20)
        return [r.to_dict() for r in all_runs[start:end]]


    def _move_to_completed(self, run: RunState) -> None:
        self._active.pop(run.run_id, None)
        self._completed[run.run_id] = run
        self._evict_old_completed()

    def _evict_old_completed(self) -> None:
        if len(self._completed) <= MAX_COMPLETED:
            return
        now = time.time()
        expired = [
            rid
            for rid, run in self._completed.items()
            if run.completed_at and now - run.completed_at > COMPLETED_TTL_SEC
        ]
        for rid in expired:
            self._completed.pop(rid, None)
        # If still over limit, remove oldest
        while len(self._completed) > MAX_COMPLETED:
            oldest_key = min(
                self._completed, key=lambda k: self._completed[k].started_at
            )
            self._completed.pop(oldest_key)
