from __future__ import annotations

import json
import subprocess

from typer.testing import CliRunner

from rune.bench.aa_manifest import (
    build_aa_attempt_matrix,
    build_artificial_analysis_manifest,
    validate_manifest,
)
from rune.bench.runner import _filesystem_diff, _filesystem_status, _git_diff, _snapshot_files
from rune.cli.main import app


def test_artificial_analysis_manifest_counts():
    manifest = build_artificial_analysis_manifest()

    validate_manifest(manifest)
    assert manifest["task_count"] == 358
    assert manifest["attempt_count"] == 1074

    components = {component["name"]: component for component in manifest["components"]}
    assert components["SWE-Bench-Pro-Hard-AA"]["questions"] == 150
    assert components["Terminal-Bench v2"]["questions"] == 84
    assert components["SWE-Atlas-QnA"]["questions"] == 124
    assert len(components["Terminal-Bench v2"]["excluded_task_ids"]) == 5


def test_artificial_analysis_attempt_matrix_counts():
    matrix = build_aa_attempt_matrix()

    assert len(matrix) == 1074
    assert matrix[0]["index"] == 1
    assert matrix[0]["benchmark"] == "swe-bench-pro-hard-aa"
    assert matrix[0]["attempt_index"] == 1
    assert matrix[2]["attempt_index"] == 3


def test_bench_aa_manifest_command_writes_json(tmp_path):
    output = tmp_path / "aa.json"
    result = CliRunner().invoke(app, ["bench", "aa-manifest", "--output", str(output)])

    assert result.exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    validate_manifest(data)


def test_bench_aa_matrix_command_filters_component(tmp_path):
    output = tmp_path / "matrix.json"
    result = CliRunner().invoke(
        app,
        ["bench", "aa-matrix", "--component", "terminal-bench-v2", "--output", str(output)],
    )

    assert result.exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 252
    assert {row["benchmark"] for row in data} == {"terminal-bench-v2"}


def test_bench_run_dry_run_writes_attempt_artifacts(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "run",
            "--benchmark",
            "terminal-bench-v2",
            "--task-id",
            "smoke",
            "--instruction",
            "write hello",
            "--output-dir",
            str(tmp_path),
            "--attempt-index",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    attempts = list((tmp_path / "terminal-bench-v2" / "smoke").iterdir())
    assert len(attempts) == 1
    attempt_dir = attempts[0]
    assert json.loads((attempt_dir / "completion_trace.json").read_text())["reason"] == "dry_run"
    assert json.loads((attempt_dir / "task.json").read_text())["attempt_index"] == 2
    assert (attempt_dir / "task.json").exists()
    assert (attempt_dir / "agent_config.json").exists()


def test_terminal_smoke_command_prints_harbor_commands():
    result = CliRunner().invoke(
        app,
        ["bench", "terminal-smoke", "--count", "2", "--harbor-command"],
    )

    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert "adaptive-rejection-sampler" in lines[0]
    assert "--agent-env RUNE_HARBOR_TASK_ID=adaptive-rejection-sampler" in lines[0]
    assert "benchmarks.harbor.rune_agent:RuneInstalledAgent" in lines[0]


def test_git_diff_includes_untracked_files(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "created.txt").write_text("hello\n", encoding="utf-8")

    diff = _git_diff(tmp_path)

    assert "created.txt" in diff
    assert "+hello" in diff


def test_filesystem_diff_tracks_non_git_created_files(tmp_path):
    before = _snapshot_files(tmp_path)
    (tmp_path / "regex.txt").write_text("abc\n", encoding="utf-8")

    diff = _filesystem_diff(tmp_path, before)
    status = _filesystem_status(tmp_path, before)

    assert "b/regex.txt" in diff
    assert "+abc" in diff
    assert "?? regex.txt" in status
