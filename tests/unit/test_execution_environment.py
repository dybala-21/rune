"""Command isolation must not silently fall back or inherit host credentials."""

from pathlib import Path

import pytest

from rune.capabilities.bash import BashParams
from rune.config.schema import SandboxConfig
from rune.safety.container_exec import container_argv, execute_container
from rune.safety.process_env import child_environment


def test_container_mounts_only_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path.parent / "private-state"))
    args = container_argv("echo ok", str(tmp_path), image="python:3.13-slim", name="test")
    assert args[args.index("--network") + 1] == "none"
    assert "--pull=never" in args and "--read-only" in args and "--cap-drop=ALL" in args
    assert args.count("--mount") == 1
    assert args[args.index("--mount") + 1] == f"type=bind,src={tmp_path.resolve()},dst={tmp_path.resolve()}"
    assert args[-2:] == ["-c", "echo ok"]


@pytest.mark.parametrize("path", ["/", str(Path.home())])
def test_container_rejects_broad_host_mounts(path):
    with pytest.raises(ValueError):
        container_argv("echo ok", path, image="python:3.13-slim", name="test")


def test_container_rejects_private_state_and_runtime_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        container_argv("echo ok", str(tmp_path), image="python:3.13-slim", name="test")
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "private"))
    workspace = tmp_path / "work"
    workspace.mkdir()
    with pytest.raises(ValueError):
        container_argv("echo ok", str(workspace), image="python:3.13-slim", name="test", env_names=("DOCKER_HOST",))


def test_child_credentials_require_explicit_input(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-host-key")
    monkeypatch.setenv("CUSTOM_TOKEN", "test-host-token")
    monkeypatch.setenv("PATH", "/bin")
    env = child_environment()
    assert "OPENAI_API_KEY" not in env and "CUSTOM_TOKEN" not in env
    assert env["PATH"] == "/bin"
    assert child_environment({"CUSTOM_TOKEN": "scoped-input"})["CUSTOM_TOKEN"] == "scoped-input"


@pytest.mark.asyncio
async def test_missing_container_runtime_does_not_execute_on_host(monkeypatch, tmp_path):
    monkeypatch.setattr("rune.safety.container_exec.shutil.which", lambda name: None)
    marker = tmp_path / "should-not-exist"
    result = await execute_container(BashParams(command=f"touch {marker}", cwd=str(tmp_path)), SandboxConfig(backend="container"))
    assert not result.success
    assert result.metadata["action_status"] == "not_executed"
    assert not marker.exists()
