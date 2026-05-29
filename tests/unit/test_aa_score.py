from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rune.bench.aa_manifest import TERMINAL_BENCH_V2_AA_TASKS, build_aa_attempt_matrix
from rune.bench.aa_score import score_summary
from rune.cli.main import app


def _terminal_matrix():
    return [
        row
        for row in build_aa_attempt_matrix()
        if row["benchmark"] == "terminal-bench-v2"
    ]


def _summary_row(
    task_id: str,
    *,
    reward: float | None = 1.0,
    fingerprint_gate_valid: bool | None = True,
    verifier_disabled: bool | None = False,
    trace_reason: str = "completed",
) -> dict[str, object]:
    return {
        "benchmark": "terminal-bench-v2",
        "task_id": task_id,
        "trial_name": f"{task_id}__trial",
        "reward": reward,
        "passed": reward == 1.0 if reward is not None else None,
        "verifier_disabled": verifier_disabled,
        "fingerprint_gate_valid": fingerprint_gate_valid,
        "trace_reason": trace_reason,
        "exception_type": None,
        "exception_message": None,
        "total_tokens_used": 1000,
        "input_tokens": 700,
        "cached_input_tokens": 100,
        "cache_write_tokens": 20,
        "reasoning_tokens": 50,
        "output_tokens": 200,
        "duration_ms": 10_000,
        "agent_wall_time_ms": 7_000,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_attempt(attempt_dir: Path, task_id: str) -> None:
    _write_json(
        attempt_dir / "task.json",
        {"benchmark": "terminal-bench-v2", "task_id": task_id, "attempt_index": 1},
    )
    _write_json(
        attempt_dir / "completion_trace.json",
        {"reason": "completed", "final_step": 1, "total_tokens_used": 1000},
    )
    _write_json(attempt_dir / "timing.json", {"agent_wall_time_ms": 7000})
    _write_json(attempt_dir / "fingerprint_gate.json", {"valid": True})
    _write_json(
        attempt_dir / "fingerprint.json",
        {
            "agent_variant_id": "rune-aa-terminal-v1",
            "install_fingerprint": {"install_mode": "wheelhouse"},
        },
    )
    (attempt_dir / "tool_calls.jsonl").write_text("", encoding="utf-8")
    (attempt_dir / "events.jsonl").write_text("", encoding="utf-8")
    (attempt_dir / "model_usage.jsonl").write_text(
        json.dumps(
            {
                "event": "model_usage",
                "input_tokens": 700,
                "cached_input_tokens": 100,
                "cache_write_tokens": 20,
                "reasoning_tokens": 50,
                "output_tokens": 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "patch.diff").write_text("", encoding="utf-8")
    (attempt_dir / "final_answer.txt").write_text("done", encoding="utf-8")


def test_score_summary_reports_official_ready_for_full_terminal_matrix():
    rows = [_summary_row(str(row["task_id"])) for row in _terminal_matrix()]

    report = score_summary({"rows": rows}, component="terminal-bench-v2")

    assert report["official_ready"] is True
    assert report["expected_attempts"] == 252
    assert report["scored_attempts"] == 252
    assert report["invalid_attempts"] == 0
    assert report["coverage"]["missing_task_count"] == 0
    assert report["score"]["score"] == 1.0
    assert report["score"]["score_percent"] == 100.0
    assert report["score"]["tokens"]["input_tokens"]["total"] == 176_400


def test_score_summary_rejects_missing_fingerprint_by_default():
    task_id = TERMINAL_BENCH_V2_AA_TASKS[0]
    rows = [_summary_row(task_id, fingerprint_gate_valid=None)]

    report = score_summary({"rows": rows}, component="terminal-bench-v2")

    assert report["official_ready"] is False
    assert report["scored_attempts"] == 0
    assert report["invalid_attempts"] == 1
    assert report["invalid_reasons"] == {"fingerprint_missing": 1}


def test_score_summary_accepts_missing_benchmark_when_task_is_in_component():
    task_id = TERMINAL_BENCH_V2_AA_TASKS[0]
    row = _summary_row(task_id)
    row["benchmark"] = None

    report = score_summary({"rows": [row]}, component="terminal-bench-v2")

    assert report["scored_attempts"] == 1
    assert report["invalid_attempts"] == 0


def test_score_summary_allows_exploratory_missing_fingerprint():
    task_id = TERMINAL_BENCH_V2_AA_TASKS[0]
    rows = [_summary_row(task_id, fingerprint_gate_valid=None)]

    report = score_summary(
        {"rows": rows},
        component="terminal-bench-v2",
        require_fingerprint=False,
    )

    assert report["score_mode"] == "exploratory"
    assert report["scored_attempts"] == 1
    assert report["score"]["score_percent"] == 100.0
    assert "fingerprint validation is disabled" in report["official_blockers"]


def test_score_summary_counts_token_budget_exhaustion():
    task_id = TERMINAL_BENCH_V2_AA_TASKS[0]
    rows = [_summary_row(task_id, reward=0.0, trace_reason="token_budget_exhausted")]

    report = score_summary({"rows": rows}, component="terminal-bench-v2")

    assert report["score"]["budget_exhausted_count"] == 1
    assert report["score"]["failed"] == 1


def test_bench_aa_score_command_scores_harbor_job(tmp_path):
    task_id = TERMINAL_BENCH_V2_AA_TASKS[0]
    job_dir = tmp_path / "job"
    trial_dir = job_dir / f"{task_id}__abc123"
    attempt_dir = trial_dir / "agent" / "rune" / "terminal-bench-v2" / task_id / "attempt-001"
    _write_attempt(attempt_dir, task_id)
    _write_json(
        trial_dir / "result.json",
        {
            "task_name": task_id,
            "trial_name": f"{task_id}__abc123",
            "task_id": {"path": task_id},
            "source": "terminal-bench",
            "config": {
                "trials_dir": str(job_dir),
                "verifier": {"disable": False},
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
        },
    )

    result = CliRunner().invoke(app, ["bench", "aa-score", str(job_dir)])

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["benchmark"] == "terminal-bench-v2"
    assert report["official_ready"] is False
    assert report["scored_attempts"] == 1
    assert report["score"]["score_percent"] == 100.0
    assert report["score"]["tokens"]["input_tokens"]["total"] == 700
