"""Deferred rule outcomes: weak-evidence successes wait for the user.

Only a correction may change a verdict — everything else, including
expiry and a dead aux model, must resolve to the success the outcome
would have been counted as anyway. These tests pin that asymmetry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from rune.memory import pending_outcomes as po


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))


@pytest.fixture
def updates(monkeypatch):
    """Capture update_rules_from_outcome calls instead of touching meta."""
    calls: list[tuple[str, bool]] = []

    def _capture(domain, task_success, goal="", error_message="",
                 relevant_keys=None):
        calls.append((domain, task_success))
        return 0

    import rune.memory.rule_learner as rl
    monkeypatch.setattr(rl, "update_rules_from_outcome", _capture)
    return calls


class TestRecord:
    def test_roundtrip(self):
        po.record_pending("chat", "summarise the notes", "done",
                          relevant_keys={"k1"})
        entries = po._load()
        assert len(entries) == 1
        assert entries[0]["domain"] == "chat"
        assert entries[0]["relevant_keys"] == ["k1"]

    def test_backlog_is_capped(self):
        for i in range(15):
            po.record_pending("chat", f"task {i}")
        entries = po._load()
        assert len(entries) == po._MAX_PENDING
        assert entries[-1]["goal"] == "task 14"

    def test_flag_off_skips_resolution(self, monkeypatch):
        monkeypatch.setenv("RUNE_DEFERRED_RULE_TRUTH", "0")
        po.record_pending("chat", "task")

        async def _run():
            return await po.resolve_pendings("new message")

        import asyncio
        assert asyncio.run(_run()) == 0


class TestResolve:
    @pytest.mark.asyncio
    async def test_correction_counts_as_failure(self, updates):
        po.record_pending("chat", "write the June summary")
        with patch.object(po, "_classify_corrections",
                          AsyncMock(return_value=[0])):
            n = await po.resolve_pendings("아니 그 합계 틀렸잖아")
        assert n == 1
        assert updates == [("chat", False)]
        assert po._load() == []

    @pytest.mark.asyncio
    async def test_unrelated_message_counts_as_success(self, updates):
        po.record_pending("chat", "write the June summary")
        with patch.object(po, "_classify_corrections",
                          AsyncMock(return_value=[])):
            n = await po.resolve_pendings("play some music")
        assert n == 1
        assert updates == [("chat", True)]
        assert po._load() == []

    @pytest.mark.asyncio
    async def test_classifier_failure_leaves_them_pending(self, updates):
        po.record_pending("chat", "write the June summary")
        with patch.object(po, "_classify_corrections",
                          AsyncMock(return_value=None)):
            n = await po.resolve_pendings("whatever")
        assert n == 0
        assert updates == []
        assert len(po._load()) == 1

    @pytest.mark.asyncio
    async def test_expiry_settles_as_success_without_a_model(self, updates):
        po.record_pending("chat", "old task")
        entries = po._load()
        entries[0]["created"] = (
            datetime.now(UTC) - timedelta(hours=po._TTL_HOURS + 1)
        ).isoformat()
        po._save(entries)
        with patch.object(po, "_classify_corrections",
                          AsyncMock(return_value=None)) as classify:
            n = await po.resolve_pendings("anything")
        assert n == 1
        assert updates == [("chat", True)]
        classify.assert_not_called()  # nothing live to classify
        assert po._load() == []

    @pytest.mark.asyncio
    async def test_nothing_pending_is_free(self):
        with patch.object(po, "_classify_corrections",
                          AsyncMock()) as classify:
            assert await po.resolve_pendings("hello") == 0
        classify.assert_not_called()
