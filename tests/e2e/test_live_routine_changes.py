"""A persistent routine reuses unchanged inputs and reruns changed inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.capabilities import cron
from rune.memory.store import MemoryStore
from rune.proactive.execution_store import ExecutionStore
from rune.proactive.routine import RoutinePolicy


@pytest.mark.asyncio
async def test_routine_change_detection(live_model, tmp_path, monkeypatch, request):
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
                           timeout_seconds=90, token_budget=80_000)
    job_id = store.create_cron_job(name="Status fixture", schedule="* * * * *", command=cron._pack_command(
        "", "Read status.txt and reply with its entire content exactly. Do not change files.", "", "", policy))
    job = cron._row_to_cronjob(store.get_cron_job(job_id))
    results, verdict = [], "failed"
    try:
        for index in range(3):
            if index == 2:
                source.write_text("Shipment 731: delivered")
            await cron.execute_cron_job(job)
            records = ExecutionStore(home / "data" / "routine-executions.db")
            try:
                results.append(records.get(f"cron:{job_id}:{int(clock[0] // 60)}")[1])
            finally:
                records.close()
            clock[0] += 60
        assert results[0]["success"] and results[0]["output"].strip() == "Shipment 731: ready"
        assert results[1]["status"] == "unchanged" and results[1]["reused"]
        assert results[2]["success"] and results[2]["output"].strip() == "Shipment 731: delivered"
        assert store.get_cron_job(job_id)["run_count"] == 2
        verdict = "passed"
    finally:
        store.close()
        if directory := request.config.getoption("--live-report-dir"):
            runs = []
            for result in results:
                if result.get("reused"):
                    continue
                timing = result.get("timings") or {}
                usage = timing.get("usage") or {}
                priced = usage.get("calls") == usage.get("reported_calls") and not usage.get("unpriced_calls")
                runs.append({"seconds": result.get("duration_ms", 0) / 1000,
                             "snapshot": {"timings": timing, "usage": {
                                 "total": usage.get("total_tokens"), "cost": {"usd": usage.get("cost_usd") if priced else None},
                             }}})
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{live_model[0]}-{request.node.name}.json").write_text(json.dumps({
                "provider": live_model[0], "model": live_model[1], "outcome": verdict,
                "decision_backend": get_config().llm.decision_routing.backend,
                "scope": "Real-model background routine; three ticks, including one without model execution",
                "results": results, "runs": runs, "tool_calls": calls,
            }, indent=2, ensure_ascii=False))
