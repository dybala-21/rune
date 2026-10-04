"""Proactive feedback through the API, real agent, and durable execution store."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI


@pytest.mark.asyncio
async def test_proactive_accept_dismiss_and_restart(live_model, tmp_path, monkeypatch, request):
    from rune.agent.background import BackgroundTask, run_background
    from rune.api.handlers.proactive import router
    from rune.config import get_config
    from rune.memory.store import MemoryStore
    from rune.proactive import bridge as bridge_module
    from rune.proactive import engine as engine_module
    from rune.proactive.bridge import BridgeConfig, ProactiveAgentBridge
    from rune.proactive.engine import ProactiveEngine
    from rune.proactive.execution_store import ExecutionStore

    home, work = tmp_path / "state", tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.setenv("RUNE_WORKSPACE", str(work))
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(work))
    monkeypatch.chdir(work)
    cfg = get_config()
    cfg.filesystem.allow_paths = [str(work)]
    cfg.llm.reasoning_effort = None
    cfg.llm.reasoning_efforts = {}
    cfg.llm.route_simple_queries = False
    source = "team,amount\nengineering,750\nsales,1000\nsupport,450\nengineering,250\n"
    (work / "expenses.csv").write_text(source)
    store = MemoryStore(home / "memory.db")
    records = ExecutionStore(home / "executions.db")
    engine = ProactiveEngine()
    engine.load_persisted_suggestions(store)
    monkeypatch.setattr(engine_module, "_engine", engine)
    monkeypatch.setattr("rune.memory.store.get_memory_store", lambda: store)
    results = []

    async def execute(goal, *, verification=None):
        result = await run_background(BackgroundTask(
            goal=goal, source="proactive", workspace=str(work),
            verification=list(verification or []), max_steps=8,
            timeout_seconds=90, token_budget=30_000,
        ))
        results.append(result)
        return result

    bridge = ProactiveAgentBridge(engine, execute, BridgeConfig(poll_interval_seconds=3600), execution_store=records)
    monkeypatch.setattr(bridge_module, "_bridge", bridge)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    outcome = "failed"
    bridge.start()
    try:
        suggestions = await engine.evaluate({"hints": [
            {"title": "Expense total", "description": "expenses.csv의 amount 합계를 읽고 숫자만 답해줘. 파일을 변경하지 마.", "confidence": .95},
            {"title": "Dismissed task", "description": "rejected.txt에 실행됨이라고 저장해줘.", "confidence": .95},
        ]})
        accepted, dismissed = suggestions[:2]
        assert not results
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 54321))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            pending = (await client.get("/api/v1/proactive/suggestions")).json()["pendingSuggestions"]
            assert {accepted.id, dismissed.id} <= {s["id"] for s in pending}
            for sid, response in ((dismissed.id, "dismiss"), (accepted.id, "accept"), (accepted.id, "accept")):
                reply = await client.post("/api/v1/proactive/feedback", json={"suggestionId": sid, "response": response})
                assert reply.status_code == 200, reply.text
            async with asyncio.timeout(100):
                while not engine.get_suggestion(accepted.id).execution_status:
                    await asyncio.sleep(.1)
            detail = (await client.get(f"/api/v1/proactive/suggestions/{accepted.id}")).json()
            assert detail["response"] == "accepted"
            assert detail["executionStatus"] in ("success", "completed"), detail
            assert detail["result"]["output"].strip().replace(",", "") == "2450", detail
        assert len(results) == 1 and results[0]["success"]
        assert (work / "expenses.csv").read_text() == source
        assert not (work / "rejected.txt").exists()
        assert records.get(dismissed.id) is None
        assert any(model.endswith(live_model[1]) for model in results[0]["timings"]["usage"]["by_model"])
        bridge.stop()
        restored = ProactiveEngine()
        restored.load_persisted_suggestions(store)
        replacement = ProactiveAgentBridge(restored, execute, execution_store=records)
        try:
            await replacement.execute_suggestion(restored.get_suggestion(accepted.id), force=True)
            await replacement._dispatch_accepted()
            assert len(results) == 1
            assert restored.get_suggestion(dismissed.id).status == "dismissed"
            assert restored.get_suggestion(accepted.id).execution_result["output"].strip() == "2450"
        finally:
            replacement.stop()
        outcome = "passed"
    finally:
        bridge.stop()
        await asyncio.sleep(0)
        directory = request.config.getoption("--live-report-dir")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{live_model[0]}-proactive.json").write_text(json.dumps({
                "provider": live_model[0], "model": live_model[1], "outcome": outcome,
                "scope": "Synthetic context hints; real generation pipeline, feedback API, bridge, configured model and persistent stores.",
                "runs": results,
            }, ensure_ascii=False, indent=2))
        records.close()
        store.close()
