"""Coalesce missed schedules and retain pending occurrences across restarts."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from weakref import WeakKeyDictionary

from rune.config.loader import get_config
from rune.daemon.heartbeat import HeartbeatScheduler
from rune.proactive.routine import task_fingerprint
from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)
_slots: WeakKeyDictionary = WeakKeyDictionary()


@dataclass
class _ExecutionSlots:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    active: set = field(default_factory=set)
    users: int = 0


@asynccontextmanager
async def execution_slot():
    loop = asyncio.get_running_loop()
    state = _slots.setdefault(loop, _ExecutionSlots())
    condition, active = state.condition, state.active
    task = asyncio.current_task()
    state.users += 1
    try:
        async with condition:
            await condition.wait_for(lambda: not get_config().cron_execution.enabled
                                     or len(active) < get_config().cron_execution.max_concurrent)
            enabled = get_config().cron_execution.enabled
            if enabled:
                active.add(task)
        yield enabled
    finally:
        async with condition:
            active.discard(task)
            state.users -= 1
            condition.notify_all()
            if not state.users:
                _slots.pop(loop, None)


class CronDispatcher:
    def __init__(self, path: Path | None = None) -> None:
        self._db = sqlite3.connect(path or rune_data() / "schedule.db", isolation_level=None)
        self._db.execute("PRAGMA busy_timeout=1000")
        self._db.execute("""CREATE TABLE IF NOT EXISTS schedules (
            job TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, cursor INTEGER NOT NULL, pending INTEGER)""")
        self._active: dict[str, asyncio.Task] = {}
        self._stopped = False
        self._failed: set[str] = set()

    async def tick(self) -> None:
        from rune.capabilities.cron import get_active_cron_jobs

        if self._stopped:
            return
        config = get_config().cron_execution
        self._failed.clear()
        now = int(time.time() // 60)
        jobs = {job.id: job for job in get_active_cron_jobs()}
        rows = {row[0]: row[1:] for row in self._db.execute("SELECT * FROM schedules")}
        for removed in rows.keys() - jobs.keys():
            self._db.execute("DELETE FROM schedules WHERE job=?", (removed,))
        for job in jobs.values():
            fingerprint = task_fingerprint(job)
            saved, cursor, pending = rows.get(job.id, (fingerprint, now - 1, None))
            if saved != fingerprint:
                cursor, pending = now - 1, None
            if not config.enabled:
                cursor, pending = now, None
            elif now >= cursor:
                first = max(cursor + 1, now - config.misfire_grace_seconds // 60)
                for minute in range(now, first - 1, -1):
                    if HeartbeatScheduler._matches_cron(job.schedule, datetime.fromtimestamp(minute * 60)):
                        pending = minute
                        break
                cursor = now
                if pending is not None and now - pending > config.misfire_grace_seconds // 60:
                    log.warning("cron_occurrence_expired", job_id=job.id, minute=pending)
                    pending = None
            self._db.execute("INSERT OR REPLACE INTO schedules VALUES (?, ?, ?, ?)",
                             (job.id, fingerprint, cursor, pending))
        self._dispatch(jobs)

    def _dispatch(self, jobs) -> None:
        config = get_config().cron_execution
        if self._stopped or not config.enabled:
            return
        now = int(time.time() // 60)
        for job_id, minute, fingerprint in self._db.execute(
            "SELECT job, pending, fingerprint FROM schedules WHERE pending IS NOT NULL ORDER BY pending, job"
        ).fetchall():
            if len(self._active) >= config.max_concurrent:
                break
            if job_id not in self._active and job_id not in self._failed and job_id in jobs:
                if (now - minute > config.misfire_grace_seconds // 60
                        or fingerprint != task_fingerprint(jobs[job_id])):
                    self._db.execute("UPDATE schedules SET pending=NULL WHERE job=?", (job_id,))
                    continue
                task = asyncio.create_task(self._execute(jobs[job_id], minute))
                self._active[job_id] = task
                task.add_done_callback(self._finished)

    async def _execute(self, job, minute: int) -> None:
        from rune.capabilities.cron import execute_cron_job, get_active_cron_jobs

        try:
            await execute_cron_job(job, scheduled_minute=minute)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("cron_dispatch_failed", job_id=job.id)
            self._failed.add(job.id)
        else:
            self._db.execute("UPDATE schedules SET pending=NULL WHERE job=? AND pending=?", (job.id, minute))
        finally:
            self._active.pop(job.id, None)
        self._dispatch({job.id: job for job in get_active_cron_jobs()})

    @staticmethod
    def _finished(task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("cron_dispatch_failed", error=str(error))

    async def stop(self) -> None:
        self._stopped = True
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._db.close()
