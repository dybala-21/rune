"""Tests for post-edit auto-verification (detect + run)."""

from __future__ import annotations

import pytest

from rune.agent.auto_verify import (
    detect_test_command,
    detect_verify_command,
    run_verify,
)


class TestDetectTestCommand:
    """Correctness test detection (distinct from the lint/typecheck detector)."""

    def test_pytest_via_tests_dir(self, tmp_path):
        (tmp_path / "tests").mkdir()
        cmd = detect_test_command(str(tmp_path))
        assert cmd is not None and "pytest" in cmd

    def test_pytest_via_test_file(self, tmp_path):
        (tmp_path / "test_thing.py").write_text("def test_x(): pass")
        cmd = detect_test_command(str(tmp_path))
        assert cmd is not None and "pytest" in cmd

    def test_pytest_uses_running_interpreter_not_bare_python(self, tmp_path):
        # Regression: a bare "python" fails to spawn on python3-only machines, so
        # run_verify returns "skip" and the verifier falls back to the Evidence
        # Gate. The command must use sys.executable (the interpreter RUNE runs
        # under, which has pytest).
        import os
        import sys

        (tmp_path / "test_thing.py").write_text("def test_x(): pass")
        cmd = detect_test_command(str(tmp_path))
        assert cmd[0] == sys.executable
        assert cmd[0] != "python"
        assert os.path.exists(cmd[0])  # actually spawnable

    def test_npm_test_script(self, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        cmd = detect_test_command(str(tmp_path))
        assert cmd is not None and "npm" in cmd

    def test_npm_placeholder_test_ignored(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}'
        )
        assert detect_test_command(str(tmp_path)) is None

    def test_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RUNE_AUTO_VERIFY_CMD", "pytest -q tests/unit")
        assert detect_test_command(str(tmp_path)) == ["pytest", "-q", "tests/unit"]

    def test_none_when_no_tests(self, tmp_path):
        (tmp_path / "solution.py").write_text("x = 1")  # source, no tests
        assert detect_test_command(str(tmp_path)) is None


def test_detect_python_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    cmd = detect_verify_command(str(tmp_path))
    assert cmd is not None and "ruff" in cmd


def test_detect_node_project(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    cmd = detect_verify_command(str(tmp_path))
    assert cmd is not None and "npm" in cmd


def test_detect_none_when_no_marker(tmp_path):
    assert detect_verify_command(str(tmp_path)) is None


@pytest.mark.asyncio
async def test_run_verify_pass(tmp_path):
    state, ev = await run_verify(["true"], str(tmp_path))
    assert state == "pass"
    assert ev == ""


@pytest.mark.asyncio
async def test_run_verify_fail_carries_evidence(tmp_path):
    state, ev = await run_verify(["sh", "-c", "echo problem here; exit 1"], str(tmp_path))
    assert state == "fail"
    assert "problem here" in ev


@pytest.mark.asyncio
async def test_run_verify_skip_on_spawn_error(tmp_path):
    state, _ = await run_verify(["this_cmd_does_not_exist_xyz"], str(tmp_path))
    assert state == "skip"
