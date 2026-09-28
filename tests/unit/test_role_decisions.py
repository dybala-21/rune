"""File-role decisions survive corrections, but never outlive their request."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rune.agent.litellm_adapter import LiteLLMAgent
from rune.agent.role_decisions import RoleDecisions

GOAL = "Read source.csv and create summary.csv."
ROLES = {"source.csv": "input", "summary.csv": "output"}


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr("rune.agent.litellm_adapter._resolve_litellm_model", lambda model: (model, {}))
    monkeypatch.setattr("rune.agent.litellm_adapter._clamp_max_tokens", lambda model, count: count)
    return LiteLLMAgent(model="test/model", tools=[], max_tokens=256)


async def test_correction_reuses_roles_and_checks_current_file_state(agent, monkeypatch, tmp_path):
    from rune.agent.postconditions import check

    classify = AsyncMock(return_value=ROLES)
    monkeypatch.setattr("rune.agent.provenance.classify_roles", classify)
    source = tmp_path / "source.csv"
    source.write_text("amount\n1\n")
    decisions = RoleDecisions(GOAL, str(tmp_path))
    async with agent.run_stream(GOAL, workspace_root=str(tmp_path), role_decisions=decisions) as stream:
        await stream._classify_artifact_roles()
        conditions = stream._postconditions
        assert check(conditions, tmp_path) == []
    source.unlink()
    async with agent.run_stream(GOAL, workspace_root=str(tmp_path), role_decisions=decisions) as stream:
        await stream._classify_artifact_roles()
        ledger = stream._ledger()
        ledger.record_read("source.csv", False)
        assert ledger.unresolved() == ["source.csv"]
        assert check(conditions, tmp_path)
    assert classify.await_count == 1


@pytest.mark.parametrize("change", ["request", "workspace", "new_turn"])
async def test_decisions_do_not_cross_request_boundaries(agent, monkeypatch, tmp_path, change):
    classify = AsyncMock(return_value=ROLES)
    monkeypatch.setattr("rune.agent.provenance.classify_roles", classify)
    decisions = RoleDecisions(GOAL, str(tmp_path))
    decisions.remember(ROLES)
    goal = GOAL if change != "request" else "Read summary.csv and create source.csv."
    root = str(tmp_path / "other") if change == "workspace" else str(tmp_path)
    if change == "new_turn":
        decisions = RoleDecisions(goal, root)
    async with agent.run_stream(goal, workspace_root=root, role_decisions=decisions) as stream:
        await stream._classify_artifact_roles()
    assert classify.await_count == 1


@pytest.mark.parametrize("exit_kind", ["normal", "cancel", "error"])
async def test_stream_exit_cancels_and_awaits_pending_roles(agent, monkeypatch, tmp_path, exit_kind):
    started, stopped = asyncio.Event(), asyncio.Event()

    async def classify(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr("rune.agent.provenance.classify_roles", classify)
    error = None
    try:
        async with agent.run_stream(GOAL, workspace_root=str(tmp_path)) as stream:
            stream._start_artifact_role_classification()
            task = stream._artifact_roles_task
            await started.wait()
            if exit_kind == "cancel":
                raise asyncio.CancelledError()
            if exit_kind == "error":
                raise RuntimeError("stream failed")
    except (asyncio.CancelledError, RuntimeError) as exc:
        error = type(exc)
    assert stopped.is_set() and task.done()
    assert error is {"normal": None, "cancel": asyncio.CancelledError, "error": RuntimeError}[exit_kind]


async def test_failed_role_lookup_is_not_cached_as_a_decision(agent, monkeypatch, tmp_path):
    classify = AsyncMock(side_effect=[{}, ROLES])
    monkeypatch.setattr("rune.agent.provenance.classify_roles", classify)
    decisions = RoleDecisions(GOAL, str(tmp_path))
    for _ in range(2):
        async with agent.run_stream(GOAL, workspace_root=str(tmp_path), role_decisions=decisions) as stream:
            await stream._classify_artifact_roles()
    assert classify.await_count == 2
    assert decisions.roles == ROLES


async def test_cancelling_a_waiting_tool_closes_the_shielded_lookup(agent, monkeypatch, tmp_path):
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def classify(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr("rune.agent.provenance.classify_roles", classify)

    async def request():
        async with agent.run_stream(GOAL, workspace_root=str(tmp_path)) as stream:
            stream._start_artifact_role_classification()
            await stream._classify_artifact_roles()

    task = asyncio.create_task(request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
