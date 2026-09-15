"""Shell writes and file tools must use the same resolved-path policy."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rune.agent.tool_adapter import _validate_with_guardian
from rune.capabilities.bash import BashParams, bash_execute
from rune.capabilities.file import FileWriteParams, file_write
from rune.safety.approval_context import approval_granted
from rune.safety.guardian import Guardian
from rune.safety.shell_writes import shell_write_targets


@pytest.mark.parametrize("command", [
    "printf data > report.txt", "printf data >> report.txt", "tee report.txt",
    "cat > report.txt <<'EOF'\nreport body\nEOF", "bash -c 'printf data > report.txt'",
    "mkdir output", "touch report.txt", "cp source.txt report.txt", "mv source.txt report.txt",
    "sh -lc 'printf data > report.txt'", "env bash -c 'printf data > report.txt'",
])
def test_relative_writes_obey_protected_cwd(command):
    guardian = Guardian()
    assert not guardian.validate_file_path("/var/tmp/report.txt").allowed
    assert not guardian.validate(command, cwd="/var/tmp").allowed
    assert _validate_with_guardian("bash_execute", {"command": command, "cwd": "/var/tmp"}).blocked


def test_shell_scopes_redirections_and_data(tmp_path):
    directory = tmp_path / "sub"
    script = f"(cd {directory} && printf hi > inner.txt); printf bye > outer.txt"
    assert shell_write_targets(script, str(tmp_path)).paths == {
        str(directory / "inner.txt"), str(tmp_path / "outer.txt"),
    }
    heredoc = "cat > out.txt <<'EOF'\ncd /etc; tee /etc/hosts\nEOF"
    assert shell_write_targets(heredoc, str(tmp_path)).paths == {str(tmp_path / "out.txt")}
    assert shell_write_targets("echo hi 2>&1 > /dev/null", str(tmp_path)).paths == set()
    assert shell_write_targets("cd sub > before.txt", str(tmp_path)).paths == {str(tmp_path / "before.txt")}


@pytest.mark.parametrize("command", ['echo hi > "$TARGET"', "echo hi > *.txt", 'cd "$TARGET"; echo hi > report.txt'])
def test_unresolved_write_targets_require_approval(command, tmp_path):
    assert Guardian().validate(command, cwd=str(tmp_path)).requires_approval


def test_symlink_target_uses_file_policy(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to("/etc/hosts")
    assert not Guardian().validate("printf hi > alias", cwd=str(tmp_path)).allowed


@pytest.mark.asyncio
async def test_denied_shell_never_starts_a_process(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr("rune.capabilities.bash._execute_oneshot", execute)
    result = await bash_execute(BashParams(command="printf hi > report.txt", cwd="/var/tmp"))
    assert not result.success
    assert result.metadata["action_status"] == "not_executed"
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_file_and_shell_config_writes_share_approval(monkeypatch, tmp_path):
    guardian = Guardian()
    guardian._home = str(tmp_path)
    monkeypatch.setattr("rune.safety.guardian.get_guardian", lambda: guardian)
    monkeypatch.setattr("rune.capabilities.file.get_guardian", lambda: guardian)
    monkeypatch.setattr("rune.capabilities.bash.get_guardian", lambda: guardian)
    path = tmp_path / ".rune" / "config.yaml"
    check = guardian.validate_file_path(str(path))
    assert check.allowed and check.requires_approval
    for tool, params in [
        ("file_write", {"path": str(path)}),
        ("bash_execute", {"command": 'echo config > "$HOME/.rune/config.yaml"', "cwd": str(tmp_path)}),
    ]:
        assert _validate_with_guardian(tool, params).requires_approval
    denied = await file_write(FileWriteParams(path=str(path), content="name: rune\n"))
    assert not denied.success and denied.metadata["requires_approval"]
    assert not path.exists()
    with approval_granted():
        written = await file_write(FileWriteParams(path=str(path), content="name: rune\n"))
    assert written.success
    assert path.read_text() == "name: rune\n"


@pytest.mark.asyncio
async def test_allowed_shell_write_produces_the_file(tmp_path):
    result = await bash_execute(BashParams(command="printf report > report.txt", cwd=str(tmp_path)))
    assert result.success
    assert (tmp_path / "report.txt").read_text() == "report"
