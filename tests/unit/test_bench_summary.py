from __future__ import annotations

import json

from typer.testing import CliRunner

from rune.bench.summary import format_summary_csv, summarize_paths
from rune.cli.main import app


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_attempt(
    attempt_dir,
    benchmark="terminal-bench-v2",
    task_id="cancel-async-tasks",
    *,
    completion=True,
):
    _write_json(
        attempt_dir / "task.json",
        {"benchmark": benchmark, "task_id": task_id, "attempt_index": 1},
    )
    if completion:
        _write_json(
            attempt_dir / "completion_trace.json",
            {
                "reason": "completed",
                "final_step": 1,
                "total_tokens_used": 1234,
                "evidence_score": 0.0,
            },
        )
        _write_json(attempt_dir / "timing.json", {"agent_wall_time_ms": 2500})
    _write_json(
        attempt_dir / "fingerprint.json",
        {
            "agent_variant_id": "rune-aa-terminal-v1",
            "install_fingerprint": {
                "install_mode": "wheelhouse",
                "wheelhouse_sha256": "abc123",
                "source_git_sha": "deadbeef",
                "source_diff_sha256": "diff123",
            },
        },
    )
    _write_json(attempt_dir / "fingerprint_gate.json", {"valid": True})
    (attempt_dir / "tool_calls.jsonl").write_text(
        json.dumps({"event": "tool_call", "name": "exec"}) + "\n"
        + json.dumps({"event": "tool_call", "name": "exec"}) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "model_usage.jsonl").write_text(
        json.dumps(
            {
                "event": "model_usage",
                "input_tokens": 900,
                "cached_input_tokens": 100,
                "cache_write_tokens": 25,
                "reasoning_tokens": 50,
                "output_tokens": 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "events.jsonl").write_text("", encoding="utf-8")
    (attempt_dir / "patch.diff").write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")
    (attempt_dir / "final_answer.txt").write_text("done", encoding="utf-8")


def test_summarize_harbor_job_with_rune_attempt(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "cancel-async-tasks__abc123"
    attempt_dir = (
        trial_dir
        / "agent"
        / "rune"
        / "terminal-bench-v2"
        / "cancel-async-tasks"
        / "attempt-001"
    )
    _write_attempt(attempt_dir)
    _write_json(
        trial_dir / "result.json",
        {
            "task_name": "cancel-async-tasks",
            "trial_name": "cancel-async-tasks__abc123",
            "task_id": {"path": "cancel-async-tasks"},
            "source": "terminal-bench",
            "config": {
                "trials_dir": str(job_dir),
                "verifier": {"disable": False},
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
            "started_at": "2026-05-18T15:48:24.000000Z",
            "finished_at": "2026-05-18T15:48:34.000000Z",
            "environment_setup": {
                "started_at": "2026-05-18T15:48:24.000000Z",
                "finished_at": "2026-05-18T15:48:25.500000Z",
            },
            "agent_setup": {
                "started_at": "2026-05-18T15:48:25.500000Z",
                "finished_at": "2026-05-18T15:48:27.000000Z",
            },
            "agent_execution": {
                "started_at": "2026-05-18T15:48:27.000000Z",
                "finished_at": "2026-05-18T15:48:31.000000Z",
            },
            "verifier": {
                "started_at": "2026-05-18T15:48:31.000000Z",
                "finished_at": "2026-05-18T15:48:34.000000Z",
            },
        },
    )
    _write_json(job_dir / "result.json", {"stats": {"n_completed_trials": 1}})

    summary = summarize_paths([job_dir])

    assert summary["count"] == 1
    assert summary["passed"] == 1
    assert summary["mean_reward"] == 1.0
    assert summary["total_tokens_used"] == 1234
    assert summary["total_input_tokens"] == 900
    assert summary["total_cached_input_tokens"] == 100
    assert summary["total_cache_write_tokens"] == 25
    assert summary["total_reasoning_tokens"] == 50
    assert summary["total_output_tokens"] == 200
    assert summary["total_tool_calls"] == 2
    assert summary["fingerprint_invalid_count"] == 0
    row = summary["rows"][0]
    assert row["benchmark"] == "terminal-bench-v2"
    assert row["task_id"] == "cancel-async-tasks"
    assert row["input_tokens"] == 900
    assert row["cached_input_tokens"] == 100
    assert row["cache_write_tokens"] == 25
    assert row["reasoning_tokens"] == 50
    assert row["output_tokens"] == 200
    assert row["duration_ms"] == 10000
    assert row["environment_setup_ms"] == 1500
    assert row["fingerprint_gate_valid"] is True
    assert row["agent_variant_id"] == "rune-aa-terminal-v1"
    assert row["install_mode"] == "wheelhouse"
    assert row["source_git_sha"] == "deadbeef"
    assert row["source_diff_sha256"] == "diff123"
    assert row["attempt_dir"] == str(attempt_dir)


def test_summarize_harbor_job_ignores_downloaded_artifact_mirror(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "cancel-async-tasks__abc123"
    attempt_dir = (
        trial_dir
        / "agent"
        / "rune"
        / "terminal-bench-v2"
        / "cancel-async-tasks"
        / "attempt-001"
    )
    artifact_attempt_dir = (
        trial_dir
        / "artifacts"
        / "rune"
        / "terminal-bench-v2"
        / "cancel-async-tasks"
        / "attempt-001"
    )
    _write_attempt(attempt_dir)
    _write_attempt(artifact_attempt_dir)
    _write_json(
        trial_dir / "result.json",
        {
            "task_name": "cancel-async-tasks",
            "trial_name": "cancel-async-tasks__abc123",
            "task_id": {"path": "cancel-async-tasks"},
            "source": "terminal-bench",
            "config": {
                "trials_dir": str(job_dir),
                "verifier": {"disable": False},
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        },
    )

    summary = summarize_paths([job_dir])

    assert summary["count"] == 1
    assert summary["rows"][0]["attempt_dir"] == str(attempt_dir)


def test_summarize_harbor_timeout_with_fingerprint_only_attempt(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "crack-7z-hash__abc123"
    attempt_dir = (
        trial_dir
        / "agent"
        / "rune"
        / "terminal-bench-v2"
        / "crack-7z-hash"
        / "attempt-001"
    )
    _write_attempt(attempt_dir, task_id="crack-7z-hash", completion=False)
    _write_json(
        trial_dir / "result.json",
        {
            "task_name": "crack-7z-hash",
            "trial_name": "crack-7z-hash__abc123",
            "task_id": {"path": str(tmp_path / "cache" / "hash" / "crack-7z-hash")},
            "source": "terminal-bench",
            "config": {
                "trials_dir": str(job_dir),
                "verifier": {"disable": False},
            },
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": {
                "type": "AgentTimeoutError",
                "message": "Agent execution timed out after 900.0 seconds",
            },
        },
    )

    summary = summarize_paths([job_dir])

    assert summary["count"] == 1
    assert summary["failed"] == 1
    row = summary["rows"][0]
    assert row["task_id"] == "crack-7z-hash"
    assert row["attempt_dir"] == str(attempt_dir)
    assert row["fingerprint_gate_valid"] is True
    assert row["trace_reason"] is None
    assert row["exception_type"] == "AgentTimeoutError"


def test_summarize_direct_attempt_runs_audit(tmp_path):
    attempt_dir = tmp_path / "attempt-001"
    _write_attempt(attempt_dir, benchmark="swe-atlas-qna", task_id="django__django-11133")

    summary = summarize_paths([attempt_dir])

    assert summary["count"] == 1
    assert summary["audit_high_severity_count"] == 1
    assert summary["rows"][0]["audit_finding_count"] == 1


def test_format_summary_csv_includes_rows(tmp_path):
    attempt_dir = tmp_path / "attempt-001"
    _write_attempt(attempt_dir)

    payload = format_summary_csv(summarize_paths([attempt_dir]))

    assert "benchmark,task_id,trial_name,reward" in payload
    assert "terminal-bench-v2,cancel-async-tasks" in payload


def test_bench_summarize_command_prints_json(tmp_path):
    attempt_dir = tmp_path / "attempt-001"
    _write_attempt(attempt_dir)

    result = CliRunner().invoke(app, ["bench", "summarize", str(attempt_dir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["rows"][0]["tool_call_count"] == 2
