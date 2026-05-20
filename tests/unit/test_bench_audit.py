from __future__ import annotations

import json

from typer.testing import CliRunner

from rune.bench.audit import audit_attempt_dir
from rune.cli.main import app


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_flags_forbidden_paths_and_git_history(tmp_path):
    _write_json(tmp_path / "task.json", {"benchmark": "terminal-bench-v2"})
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "tool_call", "args": {"command": "cat /tests/test_outputs.py"}})
        + "\n"
        + json.dumps({"event": "tool_call", "args": {"command": "git log --oneline"}})
        + "\n",
        encoding="utf-8",
    )

    result = audit_attempt_dir(tmp_path)

    rules = {finding["rule_id"] for finding in result["findings"]}
    assert "forbidden_runtime_path" in rules
    assert "git_history_mining" in rules
    assert result["high_severity_count"] == 2


def test_audit_allows_git_history_for_terminal_bench_git_tasks(tmp_path):
    _write_json(tmp_path / "task.json", {"benchmark": "terminal-bench-v2", "task_id": "fix-git"})
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "tool_call", "args": {"command": "git log --oneline"}}) + "\n",
        encoding="utf-8",
    )

    result = audit_attempt_dir(tmp_path)

    assert result["finding_count"] == 0


def test_audit_does_not_flag_project_tests_directory(tmp_path):
    _write_json(tmp_path / "task.json", {"benchmark": "terminal-bench-v2"})
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "tool_call", "args": {"path": "/app/pyknotid/tests"}}) + "\n",
        encoding="utf-8",
    )

    result = audit_attempt_dir(tmp_path)

    assert result["finding_count"] == 0


def test_audit_flags_swe_atlas_source_diff(tmp_path):
    _write_json(tmp_path / "task.json", {"benchmark": "swe-atlas-qna"})
    (tmp_path / "patch.diff").write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")

    result = audit_attempt_dir(tmp_path)

    assert result["findings"][0]["rule_id"] == "qna_source_modified"


def test_bench_audit_command_prints_json(tmp_path):
    _write_json(tmp_path / "task.json", {"benchmark": "terminal-bench-v2"})
    result = CliRunner().invoke(app, ["bench", "audit", str(tmp_path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["finding_count"] == 0
