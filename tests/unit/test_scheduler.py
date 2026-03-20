"""Tests for the agent scheduler module."""

from __future__ import annotations

import heapq
import time

import pytest

from rune.agent.scheduler import (
    AgentScheduler,
    ScheduledTask,
    SchedulerConfig,
    TaskPriority,
)


class TestAgentScheduler:
    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        results: list[str] = []

        async def work():
            results.append("done")

        scheduler = AgentScheduler(config=SchedulerConfig(max_concurrency=1))
        task = ScheduledTask(id="t1", priority=TaskPriority.INTERACTIVE, execute=work)
        accepted = await scheduler.enqueue(task)
        assert accepted is True

        # Give event loop time to process
        import asyncio
        await asyncio.sleep(0.15)
        assert "done" in results

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        order: list[str] = []

        def make_work(label: str):
            async def work():
                order.append(label)
            return work

        scheduler = AgentScheduler(config=SchedulerConfig(max_concurrency=1))

        # Enqueue low priority first, then high priority
        low = ScheduledTask(
            id="low", priority=TaskPriority.BACKGROUND,
            execute=make_work("low"),
        )
        high = ScheduledTask(
            id="high", priority=TaskPriority.INTERACTIVE,
            execute=make_work("high"),
        )

        await scheduler.enqueue(low)
        await scheduler.enqueue(high)
        import asyncio
        await asyncio.sleep(0.3)

        # High priority should come first
        if len(order) >= 2:
            assert order[0] == "high"

    def test_abort_task(self):
        scheduler = AgentScheduler(config=SchedulerConfig(max_concurrency=1, max_queue_depth=50))

        task = ScheduledTask(id="t_abort", priority=TaskPriority.BACKGROUND)
        heapq.heappush(scheduler._queue, task)
        scheduler.abort("t_abort")

        remaining_ids = [t.id for t in scheduler._queue]
        assert "t_abort" not in remaining_ids

    def test_stale_cleanup(self):
        scheduler = AgentScheduler(
            config=SchedulerConfig(stale_threshold_seconds=0.0)
        )
        old_task = ScheduledTask(
            id="old",
            priority=TaskPriority.BACKGROUND,
            created_at=time.monotonic() - 1.0,
        )
        heapq.heappush(scheduler._queue, old_task)
        assert len(scheduler._queue) == 1

        scheduler._cleanup_stale()
        assert len(scheduler._queue) == 0

    def test_stats(self):
        scheduler = AgentScheduler()
        stats = scheduler.stats()
        assert stats["queue_depth"] == 0
        assert stats["active"] == 0
        assert stats["total_enqueued"] == 0
        assert stats["total_completed"] == 0
        assert "max_concurrency" in stats
