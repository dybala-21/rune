"""Scheduled work uses the configured model and persists one bounded result."""

import pytest

from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy


@pytest.mark.asyncio
async def test_scheduled_result_survives_restart(live_model, live_report, tmp_path, monkeypatch, request):
    from rune.config import get_config

    home, work = tmp_path / "state", tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.setenv("RUNE_WORKSPACE", str(work))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(work))
    get_config().proactive.enabled = False
    store = MemoryStore(tmp_path / "tasks.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    policy = RoutinePolicy(workspace=str(work), max_steps=3, timeout_seconds=60, token_budget=20_000)
    job_id = store.create_cron_job(
        name="Arithmetic fixture", schedule="* * * * *", max_runs=1,
        command=cron._pack_command("", "173 × 29 − 417의 값을 숫자만으로 답해줘.", "", "", policy),
    )
    job = cron._row_to_cronjob(store.get_cron_job(job_id))
    outcome, runs = "failed", []
    try:
        await cron.execute_cron_job(job)
        await cron.execute_cron_job(job)
        records = ExecutionStore(home / "data" / "routine-executions.db")
        try:
            runs = records.recent(f"cron:{job_id}:")
        finally:
            records.close()
        assert store.get_cron_job(job_id)["run_count"] == 1
        assert len(runs) == 1 and not runs[0]["blocked"]
        result = runs[0]["result"]
        assert result["success"] and result["output"].strip() == "4600"
        assert result["workspace"] == str(work)
        assert result["duration_ms"] < 60_000
        assert any(model.endswith(live_model[1]) for model in result["timings"]["usage"]["by_model"])
        outcome = "passed"
    finally:
        from scripts.e2e_provenance import background_run

        live_report({"outcome": outcome, "scope": "Scheduled task and persisted result",
                     "runs": [background_run(run["result"]) for run in runs], "results": runs})
        store.close()
