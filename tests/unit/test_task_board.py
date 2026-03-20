"""Tests for the shared task board module."""

from __future__ import annotations

import pytest

from rune.agent.task_board import (
    ClaimableStatus,
    SharedTaskBoard,
    SubTask,
    SubTaskResult,
)


class TestSharedTaskBoard:
    @pytest.mark.asyncio
    async def test_claim_task(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1", description="task one"),
        ])
        claimed = await board.claim("t1", "worker-1")
        assert claimed is not None
        assert claimed.status == ClaimableStatus.CLAIMED
        assert claimed.claimed_by == "worker-1"

    @pytest.mark.asyncio
    async def test_claim_already_claimed(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1", description="task one"),
        ])
        await board.claim("t1", "worker-1")
        second = await board.claim("t1", "worker-2")
        assert second is None

    @pytest.mark.asyncio
    async def test_complete_task(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1", description="task one"),
        ])
        await board.claim("t1", "worker-1")
        await board.complete("t1", SubTaskResult(task_id="t1", success=True, output="done"))

        summary = board.summary()
        assert summary["completed"] == 1

    @pytest.mark.asyncio
    async def test_fail_propagates(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1", description="first"),
            SubTask(id="t2", description="depends on t1", dependencies=["t1"]),
        ])
        await board.claim("t1", "w1")
        await board.fail("t1", "crashed")

        summary = board.summary()
        assert summary["failed"] == 2  # t1 and t2 both failed

    @pytest.mark.asyncio
    async def test_get_ready_tasks(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1", description="first"),
            SubTask(id="t2", description="depends on t1", dependencies=["t1"]),
            SubTask(id="t3", description="independent"),
        ])
        ready = board.get_ready_tasks()
        ready_ids = {t.id for t in ready}
        assert "t1" in ready_ids
        assert "t3" in ready_ids
        assert "t2" not in ready_ids

        # Complete t1, now t2 should be ready
        await board.claim("t1", "w1")
        await board.complete("t1", SubTaskResult(task_id="t1", success=True))
        ready_after = board.get_ready_tasks()
        ready_ids_after = {t.id for t in ready_after}
        assert "t2" in ready_ids_after

    def test_circular_dependency_detected(self):
        with pytest.raises(ValueError, match="[Cc]ircular"):
            SharedTaskBoard(tasks=[
                SubTask(id="a", dependencies=["b"]),
                SubTask(id="b", dependencies=["a"]),
            ])

    @pytest.mark.asyncio
    async def test_is_done(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1"),
            SubTask(id="t2"),
        ])
        assert board.is_done() is False

        await board.claim("t1", "w")
        await board.complete("t1", SubTaskResult(task_id="t1", success=True))
        await board.claim("t2", "w")
        await board.complete("t2", SubTaskResult(task_id="t2", success=True))

        assert board.is_done() is True

    def test_summary(self):
        board = SharedTaskBoard(tasks=[
            SubTask(id="t1"),
            SubTask(id="t2"),
            SubTask(id="t3"),
        ])
        s = board.summary()
        assert s["total"] == 3
        assert s["pending"] == 3
        assert s["completed"] == 0
