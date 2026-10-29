"""Shared budgets survive process boundaries without making model requests."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from rune.agent.request_budget import BudgetExceeded
from rune.agent.spend_budget import SpendLedger
from rune.agent.timing import timed_completion
from rune.config.loader import get_config


def test_budget_survives_reopen_and_applies_daily_and_monthly_limits(tmp_path):
    path = tmp_path / "spend.db"
    first = SpendLedger(path)
    day = datetime(2026, 10, 25, tzinfo=UTC)
    request = first.reserve(.8, 1, 2, now=day)
    first.settle(request, .4)
    reopened = SpendLedger(path)
    with pytest.raises(BudgetExceeded, match="daily"):
        reopened.reserve(.7, 1, 2, now=day)
    reopened.reserve(1.1, None, 2, now=datetime(2026, 10, 26, tzinfo=UTC))
    with pytest.raises(BudgetExceeded, match="monthly"):
        reopened.reserve(.6, None, 2, now=datetime(2026, 10, 27, tzinfo=UTC))
    reopened.reserve(1, 1, 2, now=datetime(2026, 11, 1, tzinfo=UTC))


def test_concurrent_reservations_cannot_overspend(tmp_path):
    path = tmp_path / "spend.db"

    def reserve(_):
        try:
            SpendLedger(path).reserve(.2, 1, None)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(20))) == 5


def test_unknown_usage_blocks_further_spend_and_rejections_release_it(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    rejected = ledger.reserve(.9, 1, None)
    ledger.settle(rejected, None, rejected=True)
    uncertain = ledger.reserve(.9, 1, None)
    ledger.settle(uncertain, None)
    with pytest.raises(BudgetExceeded, match="unknown usage"):
        ledger.reserve(.01, 1, None)


@pytest.mark.parametrize("model", [
    "openai/gpt-6-astra", "xai/grok-4.7", "gemini/gemini-3.1-pro-preview",
    "anthropic/claude-opus-5", "typesafe/jev-1.13.0", "ollama/qwen3",
])
async def test_model_boundary_accounts_once_without_extra_calls(model):
    get_config().runtime.daily_cost_usd = 2
    completion = AsyncMock(return_value={"usage": {
        "prompt_tokens": 10, "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0},
    }})
    params = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}
    await timed_completion(completion, params)
    completion.assert_awaited_once()
    assert completion.call_args.kwargs["max_tokens"] == 32
    with sqlite3.connect(SpendLedger().path) as db:
        assert db.execute("SELECT count(*) FROM spend WHERE state='settled'").fetchone()[0] == 1


async def test_zero_budget_and_unpriced_models_are_rejected_before_dispatch():
    get_config().runtime.daily_cost_usd = 0
    completion = AsyncMock()
    with pytest.raises(BudgetExceeded):
        await timed_completion(completion, {"model": "openai/gpt-6-astra", "messages": []})
    get_config().runtime.daily_cost_usd = 1
    with pytest.raises(BudgetExceeded, match="Pricing"):
        await timed_completion(completion, {"model": "azure/unknown-deployment", "messages": []})
    completion.assert_not_awaited()


async def test_stream_is_settled_only_once():
    get_config().runtime.monthly_cost_usd = 1

    async def completion(**kwargs):
        async def stream():
            for _ in range(2):
                yield {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return stream()

    stream = await timed_completion(completion, {"model": "xai/grok-4.7", "stream": True, "messages": []})
    assert len([chunk async for chunk in stream]) == 2
    await stream.aclose()
    with sqlite3.connect(SpendLedger().path) as db:
        assert db.execute("SELECT state, amount FROM spend").fetchall() == [("settled", 32)]


async def test_cancelled_call_retains_reservation():
    get_config().runtime.daily_cost_usd = 1

    async def completion(**kwargs):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await timed_completion(completion, {"model": "xai/grok-4.7", "messages": []})
    with sqlite3.connect(SpendLedger().path) as db:
        assert db.execute("SELECT state FROM spend").fetchone()[0] == "unknown"


async def test_disabled_shared_budget_does_not_create_ledger_or_change_request():
    completion = AsyncMock(return_value={})
    params = {"model": "custom/model", "messages": []}
    await timed_completion(completion, params)
    completion.assert_awaited_once_with(**params)
    assert not SpendLedger().path.exists()


def test_pruning_details_keeps_period_totals_and_unresolved_usage(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    old = datetime(2020, 1, 1, tzinfo=UTC)
    settled = ledger.reserve(.8, 1, 2, now=old)
    ledger.settle(settled, .4)
    reserved = ledger.reserve(.2, 1, 2, now=old)
    ledger.prune()
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT id FROM spend").fetchall() == [(reserved,)]
        assert db.execute("SELECT spent, held FROM spend_periods WHERE period='2020-01-01'").fetchone() == (400000, 200000)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(.5, 1, 2, now=old)


def test_late_usage_reconciles_once_and_preserves_original_period(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    request = ledger.reserve(.8, 1, 10, now=datetime(2026, 10, 31, tzinfo=UTC))
    ledger.settle(request, None)
    ledger.settle(request, None)
    ledger.settle(request, .3)
    ledger.settle(request, .3)
    db = ledger._connect()
    try:
        assert db.execute("SELECT period, spent, held, unknown FROM spend_periods ORDER BY period").fetchall() == [
            ("2026-10", 300000, 0, 0), ("2026-10-31", 300000, 0, 0)]
        assert db.execute("SELECT evidence FROM spend_reconciliations").fetchall() == [("provider_usage",)]
    finally:
        db.close()


def test_reconciliation_requires_evidence_and_cannot_release_active_requests(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    request = ledger.reserve(.8, 1, 10)
    with pytest.raises(ValueError, match="Only requests"):
        ledger.reconcile(request, .2, "invoice reference")
    ledger.settle(request, None)
    with pytest.raises(ValueError, match="reference"):
        ledger.reconcile(request, .2, " ")
    with pytest.raises(ValueError, match="finite"):
        ledger.reconcile(request, float("nan"), "invoice reference")
    ledger.reconcile(request, .2, "invoice reference")
    assert ledger.unresolved() == []
    ledger.reserve(.8, 1, 10)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(.01, 1, 10)


def test_reconciling_one_request_does_not_clear_other_unknown_usage(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    first, second = [ledger.reserve(.2, 1, 10) for _ in range(2)]
    ledger.settle(first, None)
    ledger.settle(second, None)
    ledger.reconcile(first, .1, "invoice reference")
    assert [row["id"] for row in ledger.unresolved()] == [second]
    with pytest.raises(BudgetExceeded, match="unknown usage"):
        ledger.reserve(.1, 1, 10)


def test_crashed_process_can_be_reconciled_without_releasing_live_reservations(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "spend.db"
    source = """
import sys
from pathlib import Path
from rune.agent.spend_budget import SpendLedger
ledger = SpendLedger(Path(sys.argv[1]))
print(ledger.reserve(.8, 1, 10), flush=True)
sys.stdin.read()
"""
    child = subprocess.Popen([sys.executable, "-c", source, str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        request = child.stdout.readline().strip()
        assert request
        ledger = SpendLedger(path)
        assert ledger.recover() == 0
        with pytest.raises(ValueError, match="Only requests"):
            ledger.reconcile(request, .05, "confirmed usage")
        child.kill()
        child.wait(timeout=5)
        assert ledger.recover() == 1
        assert ledger.recover() == 0
        assert ledger.unresolved()[0]["state"] == "unknown"
        with pytest.raises(BudgetExceeded, match="unknown usage"):
            ledger.reserve(.01, 1, 10)
        ledger.reconcile(request, .05, "confirmed usage")
        ledger.reserve(.9, 1, 10)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT spent, held, unknown FROM spend_periods").fetchall() == [
                (50000, 900000, 0), (50000, 900000, 0)]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()


def test_untracked_old_reservations_are_not_assumed_dead(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    request = ledger.reserve(.8, 1, 10)
    with sqlite3.connect(ledger.path) as db:
        db.execute("DELETE FROM spend_owners")
    assert ledger.recover() == 0
    with pytest.raises(ValueError, match="Only requests"):
        ledger.reconcile(request, 0, "unconfirmed owner")


def test_owner_retention_keeps_live_and_unresolved_leases(tmp_path):
    import os
    from uuid import uuid4

    from rune.agent.spend_owner import owner_id, prune

    database = tmp_path / "spend.db"
    active = owner_id(database)
    directory = database.with_suffix(".owners")
    retained, expired = uuid4().hex, uuid4().hex
    for name in (retained, expired, "unrecognized"):
        (directory / name).touch()
    for path in directory.iterdir():
        os.utime(path, (1, 1))
    prune(database, {retained})
    assert {path.name for path in directory.iterdir()} == {active, retained, "unrecognized"}
