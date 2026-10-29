"""Test startup deadlines, connection failures, and saved daemon addresses."""

import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
import typer

from rune.daemon.connections import Connections
from rune.daemon.lifecycle import Cleanup, IncompleteShutdown, start_until_stopped


async def test_startup_cannot_wait_forever_for_cancelled_work(monkeypatch):
    from rune.daemon import lifecycle

    monkeypatch.setattr(lifecycle, "STARTUP_TIMEOUT", .01)
    monkeypatch.setattr(lifecycle, "CANCEL_TIMEOUT", .01)
    release, cancelled = asyncio.Event(), asyncio.Event()

    async def stalled():
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    try:
        with pytest.raises(IncompleteShutdown):
            await asyncio.wait_for(start_until_stopped(stalled, asyncio.Event()), .5)
        assert cancelled.is_set()
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_still_running_startup_retains_process_ownership(monkeypatch):
    from unittest.mock import AsyncMock

    from rune.daemon import lifecycle
    from rune.daemon.main import RuneDaemon
    from rune.daemon.process_lock import get_lock_owner

    daemon = RuneDaemon(install_signal_handlers=False)
    release = asyncio.Event()
    workers = []
    monkeypatch.setattr(lifecycle, "STARTUP_TIMEOUT", .01)
    monkeypatch.setattr(lifecycle, "CANCEL_TIMEOUT", .01)

    async def start():
        workers.append(asyncio.current_task())
        await daemon._acquire_process_lock()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    monkeypatch.setattr(daemon, "start", start)
    monkeypatch.setattr(daemon, "_shutdown_subsystems", AsyncMock())
    try:
        with pytest.raises(RuntimeError, match="ownership retained"):
            await asyncio.wait_for(daemon.serve_forever(), .5)
        assert get_lock_owner() == os.getpid()
    finally:
        release.set()
        await asyncio.gather(*workers)
        await daemon._release_process_lock()


async def test_cancel_resistant_connection_is_reported_during_cleanup():
    connections = Connections()
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def stalled():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()

    connections.start("mcp", stalled, timeout=.01)
    try:
        await entered.wait()
        await cancelled.wait()
        await connections.tasks["mcp"]
        assert connections.status == {"mcp": "failed"}
        cleanup = Cleanup(seconds=.01)
        await asyncio.wait_for(connections.stop(cleanup), .5)
        with pytest.raises(RuntimeError, match="connection_work_mcp"):
            cleanup.check()
    finally:
        release.set()
        await connections.work["mcp"]


async def test_optional_connections_do_not_wait_for_each_other():
    connections = Connections()
    release = asyncio.Event()

    async def ready():
        return "ready"

    connections.start("mcp", release.wait)
    connections.start("channels", ready)
    await connections.tasks["channels"]
    assert connections.status == {"mcp": "connecting", "channels": "ready"}
    cleanup = Cleanup()
    await connections.stop(cleanup)
    cleanup.check()


def test_foreground_process_exits_if_asyncio_shutdown_ignores_cancellation(tmp_path):
    source = """
import asyncio
from rune.daemon.lifecycle import run_foreground
class Daemon:
    async def serve_forever(self):
        self._process_deadline.arm(.15)
        async def stuck():
            while True:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    continue
        asyncio.create_task(stuck())
        await asyncio.sleep(0)
run_foreground(Daemon())
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, timeout=5)
    assert result.returncode == 1
    assert b"could not stop safely" in result.stderr


def test_service_and_cli_share_saved_address(monkeypatch, tmp_path):
    from rune.cli import main
    from rune.daemon import readiness, service
    from rune.daemon.endpoint import Endpoint, resolve_endpoint, save_endpoint

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.delenv("RUNE_API_HOST", raising=False)
    monkeypatch.delenv("RUNE_API_PORT", raising=False)
    endpoint = Endpoint("127.0.0.1", 19876)
    save_endpoint(endpoint)
    assert resolve_endpoint() == endpoint
    assert service.launchd_config(endpoint.host, endpoint.port)["ProgramArguments"][-4:] == [
        "--host", "127.0.0.1", "--port", "19876"]
    assert '--host "127.0.0.1" --port 19876' in service.systemd_config(endpoint.host, endpoint.port)
    monkeypatch.setattr(service, "get_service_status", lambda: "stopped")
    start = Mock()
    monkeypatch.setattr(service, "start_service", start)
    owners = iter([None, 123])
    monkeypatch.setattr(main, "_daemon_pid", lambda: next(owners))
    probe = Mock(return_value=True)
    monkeypatch.setattr(readiness, "is_ready", probe)
    with pytest.raises(typer.Exit) as result:
        main.daemon_start()
    assert result.value.exit_code == 0
    probe.assert_called_once_with("127.0.0.1", 19876, 123)
    start.assert_called_once()


def test_conflicting_service_port_fails_before_start(monkeypatch, tmp_path):
    from rune.cli import main
    from rune.daemon import service
    from rune.daemon.endpoint import Endpoint, save_endpoint

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    save_endpoint(Endpoint(port=19876))
    monkeypatch.setattr(main, "_daemon_pid", lambda: None)
    monkeypatch.setattr(service, "get_service_status", lambda: "stopped")
    start = Mock()
    monkeypatch.setattr(service, "start_service", start)
    with pytest.raises(typer.Exit) as result:
        main.daemon_start(host="127.0.0.1", port=19877)
    assert result.value.exit_code == 1
    start.assert_not_called()
    assert json.loads((tmp_path / "daemon.json").read_text())["port"] == 19876


def test_endpoint_file_is_private_and_rejects_bad_ports(tmp_path, monkeypatch):
    from rune.daemon.endpoint import Endpoint, save_endpoint

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    save_endpoint(Endpoint())
    assert os.stat(tmp_path / "daemon.json").st_mode & 0o777 == 0o600
    for port in (True, 0, -1, 65536):
        with pytest.raises(ValueError):
            Endpoint(port=port)
    with pytest.raises(ValueError):
        Endpoint(host=123)


def test_failed_service_install_preserves_saved_address(monkeypatch, tmp_path):
    from rune.daemon import service
    from rune.daemon.endpoint import Endpoint, save_endpoint, saved_endpoint

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    save_endpoint(Endpoint(port=19876))
    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(service, "_install_launchd", Mock(side_effect=OSError("bootstrap failed")))
    with pytest.raises(OSError):
        service.install_service(port=19877)
    assert saved_endpoint() == Endpoint(port=19876)


async def test_busy_budget_recovery_does_not_misreport_disk_health(monkeypatch):
    import sqlite3
    import time

    from rune.agent.spend_budget import SpendLedger
    from rune.daemon.maintenance import Maintenance

    monkeypatch.setattr(SpendLedger, "recover", Mock(side_effect=sqlite3.OperationalError("database is locked")))
    maintenance = Maintenance(None)
    maintenance._last_prune = time.monotonic()
    await maintenance.tick()
    assert maintenance.healthy
