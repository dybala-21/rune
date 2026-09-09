"""Background memory learning and search indexing for web runs."""

from __future__ import annotations

import asyncio
import json
import time
from contextvars import Context
from dataclasses import asdict
from typing import Any

from rune.agent.agent_context import AgentContext, PostProcessInput
from rune.api.run_store import RunStore
from rune.utils.logger import get_logger

log = get_logger(__name__)


class RunMaintenance:
    """Queue work on disk and serialize updates to shared memory state."""

    def __init__(self, store: RunStore) -> None:
        self._store = store
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        db = self._store.db
        with db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS web_run_maintenance (
                    run_id TEXT PRIMARY KEY REFERENCES web_runs(run_id) ON DELETE CASCADE,
                    payload TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            # Keep interrupted jobs for inspection: some rule updates may
            # already have been applied.
            interrupted = db.execute(
                "UPDATE web_run_maintenance SET stage = 'interrupted' WHERE stage = 'learning'"
            ).rowcount
        if interrupted:
            log.warning("run_maintenance_interrupted", count=interrupted)
        self._closing = False
        self._wake()

    def enqueue(
        self, run_id: str, context: Any, trace: Any, full_text: str,
        duration_ms: int, *, classification_hint: str | None = None,
    ) -> None:
        inp = PostProcessInput(
            context=AgentContext(
                goal=context.goal, original_goal=context.original_goal,
                conversation_id=context.conversation_id,
            ),
            success=trace.reason == "completed", answer=full_text,
            duration_ms=duration_ms, reason=trace.reason,
            verification=getattr(trace, "verification", None),
            mech_check=getattr(trace, "mech_check", ""),
            evidence_gate=getattr(trace, "evidence_gate", None),
            classification_hint=classification_hint,
        )
        with self._store.db:
            self._store.db.execute(
                "INSERT INTO web_run_maintenance (run_id, payload) VALUES (?, ?)",
                (run_id, json.dumps(asdict(inp), ensure_ascii=False)),
            )
        self._wake()

    def _wake(self) -> None:
        if not self._closing and (self._worker is None or self._worker.done()):
            # Background work must not inherit a finished run's tool context.
            self._worker = asyncio.create_task(self._drain(), name="run-maintenance", context=Context())

    async def _drain(self) -> None:
        try:
            while not self._closing:
                row = self._store.db.execute(
                    "SELECT run_id, payload, stage FROM web_run_maintenance "
                    "WHERE stage IN ('pending', 'indexing') ORDER BY rowid LIMIT 1"
                ).fetchone()
                if row is None:
                    return
                run_id, raw, stage = row
                started = time.monotonic()
                try:
                    payload = json.loads(raw)
                    if stage == "pending":
                        self._set_stage(run_id, "learning")
                        from rune.agent.agent_context import post_process_agent_result

                        inp = PostProcessInput(**{
                            **payload, "context": AgentContext(**payload["context"]),
                        })
                        async with asyncio.timeout(120):
                            await post_process_agent_result(inp)
                        self._set_stage(run_id, "indexing")

                    from rune.api.conversation_wiring import get_conv_manager

                    manager = get_conv_manager()
                    session_id = payload["context"]["conversation_id"]
                    if manager is not None and session_id:
                        # Include follow-ups saved while learning was running.
                        conversation = await manager._store.load(session_id)
                        if conversation is not None:
                            await manager._store.embed_turns(conversation.turns)
                    with self._store.db:
                        self._store.db.execute("DELETE FROM web_run_maintenance WHERE run_id = ?", (run_id,))
                    log.debug("run_maintenance_completed", run_id=run_id,
                              duration_ms=round((time.monotonic() - started) * 1000))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._set_stage(run_id, "failed")
                    log.warning("run_maintenance_failed", run_id=run_id, error=str(exc)[:200])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("run_maintenance_worker_failed", error=str(exc)[:200])

    def _set_stage(self, run_id: str, stage: str) -> None:
        with self._store.db:
            self._store.db.execute(
                "UPDATE web_run_maintenance SET stage = ? WHERE run_id = ?", (stage, run_id),
            )

    async def close(self, grace_seconds: float = 5.0) -> None:
        """Allow the current job to finish; leave queued work for the next start."""
        self._closing = True
        if self._worker is None:
            return
        _, pending = await asyncio.wait({self._worker}, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        await asyncio.gather(self._worker, return_exceptions=True)
