"""Tests for post-edit auto-verification (detect + run)."""

from __future__ import annotations

import pytest

from rune.agent.auto_verify import detect_verify_command, run_verify


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
