from __future__ import annotations

import pytest

from rune.capabilities.bash import (
    BashParams,
    bash_execute,
    blocked_benchmark_vcs_history_command,
)


def test_benchmark_vcs_guard_blocks_git_history(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_BLOCK_VCS_HISTORY", "1")

    assert blocked_benchmark_vcs_history_command("git log --oneline -20") == "git log"
    assert blocked_benchmark_vcs_history_command("git -C /app/repo show HEAD") == "git show"
    assert blocked_benchmark_vcs_history_command("FOO=1 git blame file.py") == "git blame"


def test_benchmark_vcs_guard_allows_non_history_git(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_BLOCK_VCS_HISTORY", "1")

    assert blocked_benchmark_vcs_history_command("git status --short") is None
    assert blocked_benchmark_vcs_history_command("git clone --depth 1 https://example.com/repo.git") is None


def test_benchmark_vcs_guard_can_be_disabled_for_explicit_git_tasks(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_BLOCK_VCS_HISTORY", "1")
    monkeypatch.setenv("RUNE_BENCH_ALLOW_VCS_HISTORY", "1")

    assert blocked_benchmark_vcs_history_command("git log --oneline -20") is None


@pytest.mark.asyncio
async def test_bash_execute_blocks_benchmark_vcs_history_before_execution(monkeypatch):
    monkeypatch.setenv("RUNE_BENCH_BLOCK_VCS_HISTORY", "1")

    result = await bash_execute(BashParams(command="git log --oneline -20"))

    assert result.success is False
    assert result.metadata == {"benchmark_policy_block": "vcs_history", "command": "git log"}
