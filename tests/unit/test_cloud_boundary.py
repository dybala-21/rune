"""Hosting cannot be downgraded through project config or ordinary agent tools."""

import pytest
from starlette.requests import Request

from rune.cloud.store import CloudStore
from rune.config.schema import SandboxConfig


def test_hosted_environment_ignores_local_overrides(tmp_path, monkeypatch):
    from rune.safety.execution_environment import environment_scope, execution_config
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    with environment_scope(str(tmp_path), SandboxConfig(backend="local", allow_network=True)):
        config = execution_config()
        assert config.backend == "container" and not config.allow_network
    with pytest.raises(ValueError):
        with environment_scope(str(tmp_path.parent)):
            pytest.fail("Workspace boundary was expanded")


def test_broker_and_cloud_state_cannot_be_mounted(tmp_path, monkeypatch):
    from rune.safety.container_exec import container_argv
    monkeypatch.setenv("RUNE_BROKER_HOME", str(tmp_path / "broker"))
    monkeypatch.setenv("RUNE_CLOUD_HOME", str(tmp_path / "cloud"))
    for name in ("broker", "cloud"):
        path = tmp_path / name
        path.mkdir()
        with pytest.raises(ValueError):
            container_argv("cat connectors.db", str(path), image="python:3.13-slim", name="test")


async def test_recursive_search_cannot_read_secret_symlink(tmp_path, monkeypatch):
    from rune.capabilities.file import FileSearchParams, file_search
    from rune.safety.guardian import Guardian
    broker = tmp_path / "broker"
    broker.mkdir()
    monkeypatch.setenv("RUNE_BROKER_HOME", str(broker))
    secret = broker / "connectors.db"
    secret.write_text("PRIVATE_SENTINEL")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(secret)
    guardian = Guardian()
    assert not guardian.validate_file_read_path(str(secret)).allowed
    result = await file_search(FileSearchParams(path=str(workspace), pattern="PRIVATE_SENTINEL"))
    assert result.success and not result.output


async def test_worker_requires_token_even_on_loopback(monkeypatch):
    from fastapi import HTTPException

    from rune.api.auth import TokenAuthDependency
    monkeypatch.setenv("RUNE_REQUIRE_TOKEN", "1")
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1000), "server": ("127.0.0.1", 18789)})
    with pytest.raises(HTTPException) as err:
        await TokenAuthDependency()(request)
    assert err.value.status_code == 401


def test_hosted_tools_and_terminal_cannot_escape(monkeypatch, tmp_path):
    from rune.api.terminal import is_enabled
    from rune.cloud.boundary import check_tool
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    monkeypatch.setenv("RUNE_TERMINAL_ENABLED", "1")
    assert not is_enabled()
    assert check_tool("credential_save", {})
    assert check_tool("file_read", {"path": "/etc/shadow"})
    assert check_tool("bash_execute", {"cwd": "/"})
    assert check_tool("bash_execute", {"cwd": str(tmp_path)}) is None


def test_hosted_workers_can_recall_and_tune_their_own_memory(monkeypatch, tmp_path):
    from rune.cloud.boundary import check_tool
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", str(tmp_path))
    assert check_tool("conversation_search", {"query": "dinner"}) is None
    assert check_tool("memory_tune", {"preset": "accuracy"}) is None
    for name in ("delegate_task", "document_read", "safety_tune", "service_connect"):
        assert check_tool(name, {})


async def test_hosted_authority_survives_environment_changes(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from rune.api.auth import TokenAuthDependency, token_required
    from rune.api.terminal import is_enabled
    from rune.cloud import boundary
    from rune.safety.tool_policy import approval_mode

    monkeypatch.setattr(boundary, "_HOSTED", True)
    monkeypatch.setattr(boundary, "_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "0")
    monkeypatch.setenv("RUNE_REQUIRE_TOKEN", "0")
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "bypass")
    monkeypatch.setenv("RUNE_TERMINAL_ENABLED", "1")
    monkeypatch.setenv("RUNE_ISOLATION_ROOT", "/")
    assert token_required() and not is_enabled()
    assert approval_mode() == "standard"
    assert boundary.check_path("/etc/shadow")
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1000), "server": ("127.0.0.1", 18789)})
    with pytest.raises(HTTPException) as err:
        await TokenAuthDependency()(request)
    assert err.value.status_code == 401


def test_partial_provision_never_accepts_user_traffic(tmp_path):
    store = CloudStore(tmp_path / "cloud")
    store.reserve("alice")
    token = store.register("alice", "http://10.207.0.6:18789", "rune_internal")
    for status in ("reserved", "provisioning", "failed", "stopping", "stopped", "starting"):
        store.status("alice", status)
        assert store.authenticate(token) is None
    store.status("alice", "running")
    assert store.authenticate(token)


def test_hosted_loader_ignores_project_environment_and_credentials(tmp_path, monkeypatch):
    from rune.config import loader
    from rune.utils.env import load_env
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "private"))
    project = tmp_path / ".rune"
    project.mkdir()
    (project / ".env").write_text('RUNE_MCP_SERVERS=untrusted-command\nRUNE_TEST_INJECTED=bad\n')
    (project / "config.yaml").write_text("safety:\n  sandbox:\n    backend: local\n")
    (project / "google-credentials.json").write_text('{"project_id":"untrusted"}')
    monkeypatch.delenv("RUNE_MCP_SERVERS", raising=False)
    monkeypatch.delenv("RUNE_TEST_INJECTED", raising=False)
    assert "RUNE_TEST_INJECTED" not in load_env()
    assert loader._find_config_file() is None
    assert loader._resolve_api_keys({}).get("google_credentials_file") != str(project / "google-credentials.json")
