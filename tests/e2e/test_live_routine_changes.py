"""A persistent routine reuses unchanged inputs and reruns changed inputs."""

from types import SimpleNamespace

import pytest

from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy


@pytest.mark.asyncio
async def test_routine_change_detection(live_model, live_report, tmp_path, monkeypatch, request):
    from rune.agent import loop
    from rune.config import get_config
    from rune.proactive import routine

    home, work = tmp_path / "state", tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.setenv("RUNE_WORKSPACE", str(work))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(work))
    get_config().proactive.enabled = False
    calls = []
    original_loop = loop.NativeAgentLoop

    def create_loop(config):
        agent = original_loop(config)

        async def record(call):
            calls.append(call)

        agent.on("tool_call", record)
        return agent

    monkeypatch.setattr(loop, "NativeAgentLoop", create_loop)
    clock = [1000.0]
    monkeypatch.setattr(routine, "time", SimpleNamespace(time=lambda: clock[0]))
    source = work / "status.txt"
    source.write_text("Shipment 731: ready")
    store = MemoryStore(tmp_path / "tasks.db")
    monkeypatch.setattr(cron, "_get_store", lambda: store)
    policy = RoutinePolicy(workspace=str(work), input_paths=["status.txt"], max_steps=8,
                           timeout_seconds=90, token_budget=80_000, total_model_requests=12)
    job_id = store.create_cron_job(name="Status fixture", schedule="* * * * *", command=cron._pack_command(
        "", "Read status.txt and reply with its entire content exactly. Do not change files.", "", "", policy))
    results, usage, verdict = [], [], "failed"
    try:
        for index in range(3):
            if index == 1:
                store.close()
                store = MemoryStore(tmp_path / "tasks.db")
            if index == 2:
                source.write_text("Shipment 731: delivered")
            job = cron._row_to_cronjob(store.get_cron_job(job_id))
            await cron.execute_cron_job(job)
            if index == 0:
                await cron.execute_cron_job(job)
            records = ExecutionStore(home / "data" / "routine-executions.db")
            try:
                results.append(records.get(f"cron:{job_id}:{int(clock[0] // 60)}")[1])
                usage.append(records.usage(f"cron:{job_id}:"))
            finally:
                records.close()
            clock[0] += 60
        assert results[0]["success"] and results[0]["output"].strip() == "Shipment 731: ready"
        assert results[1]["status"] == "unchanged" and results[1]["reused"]
        assert results[2]["success"] and results[2]["output"].strip() == "Shipment 731: delivered"
        assert store.get_cron_job(job_id)["run_count"] == 2
        assert usage[0] == usage[1]
        assert 0 < usage[0]["modelRequests"] < usage[2]["modelRequests"] <= policy.total_model_requests
        assert usage[0]["tokens"] < usage[2]["tokens"]
        assert all(row["requestsKnown"] and row["tokensKnown"] and row["costKnown"] for row in usage)
        verdict = "passed"
    finally:
        store.close()
        from scripts.e2e_provenance import background_run

        live_report({"outcome": verdict,
                     "scope": "Real-model routine; duplicate tick, store restart, unchanged and changed inputs",
                     "results": results, "usage": usage,
                     "runs": [background_run(r) for r in results if not r.get("reused")],
                     "tool_calls": calls})
