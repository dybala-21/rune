"""Exercise detached startup, crash accounting, and stuck shutdown through the CLI."""

import json
import os
import plistlib
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import pytest

REPO = Path(__file__).resolve().parents[2]
TOKENIZER_CACHE = Path(find_spec("litellm").origin).parent / "litellm_core_utils/tokenizers"
GUARD = """
import asyncio, os, socket
from pathlib import Path
root = Path(os.environ['RUNE_HOME'])
connect = socket.socket.connect
def local_only(self, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', 'localhost', '::1'):
        raise RuntimeError('External connections are disabled in this test')
    return connect(self, address)
socket.socket.connect = local_only
from rune.agent import timing
async def denied(*args, **kwargs):
    (root / 'unexpected-model-call').touch()
    raise AssertionError('Paid calls are disabled in this test')
if not os.environ.get('TEST_BUDGET'):
    timing.timed_completion = denied
if os.environ.get('TEST_STUCK_MCP'):
    from rune.daemon.main import RuneDaemon
    async def stuck(self):
        (root / 'mcp-started').touch()
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                (root / 'mcp-cancelled').touch()
    RuneDaemon._initialize_mcp_bridge = stuck
"""


def wait_for(check, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(.05)
    raise AssertionError("Timed out waiting for the test process")


@pytest.fixture
def runtime_root():
    # macOS Unix sockets cannot use pytest's long default temporary paths.
    with tempfile.TemporaryDirectory(prefix="rune-cli-", dir="/tmp") as directory:
        yield Path(directory)


@pytest.fixture
def runtime(runtime_root):
    tmp_path = runtime_root
    home = tmp_path / "state"
    home.mkdir()
    (tmp_path / "work").mkdir()
    (home / "config.yaml").write_text(
        "proactive:\n  enabled: false\nskills:\n  gatedLearning: false\n"
        "runtime:\n  dailyCostUsd: 1\n  monthlyCostUsd: 10\n"
    )
    (tmp_path / "sitecustomize.py").write_text(GUARD)
    env = {key: os.environ[key] for key in ("PATH", "USER", "TMPDIR", "LANG") if key in os.environ}
    env.update(HOME=str(tmp_path), RUNE_HOME=str(home), RUNE_WORKSPACE=str(tmp_path / "work"),
               RUNE_WEB_STATIC_DIR=str(REPO / "web/dist"), PYTHONPATH=f"{tmp_path}{os.pathsep}{REPO}",
               TIKTOKEN_CACHE_DIR=str(TOKENIZER_CACHE))

    def cli(*args, ok=True):
        result = subprocess.run([sys.executable, "-m", "rune.cli.main", *args],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=38)
        if ok:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    try:
        yield home, env, cli
    finally:
        cli("daemon", "stop", ok=False)
        assert not (home / "unexpected-model-call").exists()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ready(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/ready", timeout=2) as response:
        return json.load(response)


def test_custom_port_survives_detached_restart(runtime):
    home, _, cli = runtime
    port = free_port()
    started = time.monotonic()
    cli("daemon", "start", "--port", str(port))
    startup = time.monotonic() - started
    first = ready(port)
    assert first["ready"]
    assert json.loads((home / "daemon.json").read_text()) == {"host": "127.0.0.1", "port": port}
    assert (home / "daemon.json").stat().st_mode & 0o777 == 0o600
    assert "already running" in cli("daemon", "start").stderr
    assert ready(port)["pid"] == first["pid"]
    assert "ready" in cli("daemon", "status").stderr
    cli("daemon", "stop")
    assert not (home / "data/rune.pid").exists()
    cli("daemon", "start")
    second = ready(port)
    assert second["ready"] and second["pid"] != first["pid"]
    print(json.dumps({"custom_port_restart": True, "duplicate_start_reused_pid": True,
                      "cli_startup_seconds": round(startup, 3)}))


def test_crashed_usage_is_recovered_and_reconciled_through_cli(runtime):
    home, env, cli = runtime
    source = """
import asyncio, os
from pathlib import Path
from rune.agent.timing import timed_completion
root = Path(os.environ['RUNE_HOME'])
async def completion(**kwargs):
    with (root / 'stub-calls').open('a') as handle:
        handle.write('called\\n')
    if os.environ.get('TEST_HOLD'):
        await asyncio.Event().wait()
    return {'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
asyncio.run(timed_completion(completion, {'model': 'xai/grok-4.7', 'messages': [], 'max_tokens': 32}))
"""
    worker_env = {**env, "TEST_BUDGET": "1"}
    worker = subprocess.Popen([sys.executable, "-c", source], cwd=home.parent,
                              env={**worker_env, "TEST_HOLD": "1"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: (home / "stub-calls").exists())
        live = json.loads(cli("budget", "status").stdout)["unresolved"]
        assert len(live) == 1 and live[0]["state"] == "reserved"
        request = live[0]["id"]
        rejected = cli("budget", "reconcile", request, "--cost-usd", "0", "--evidence", "test fixture", ok=False)
        assert rejected.returncode != 0
        worker.kill()
        worker.wait(timeout=5)
        recovered = json.loads(cli("budget", "status").stdout)["unresolved"]
        assert len(recovered) == 1 and recovered[0]["state"] == "unknown"
        blocked = subprocess.run([sys.executable, "-c", source], cwd=home.parent, env=worker_env,
                                 capture_output=True, text=True, timeout=10)
        assert blocked.returncode != 0 and "unknown usage" in blocked.stderr
        assert (home / "stub-calls").read_text().splitlines() == ["called"]
        cli("budget", "reconcile", request, "--cost-usd", ".005", "--evidence", "isolated fixture receipt")
        result = subprocess.run([sys.executable, "-c", source], cwd=home.parent, env=worker_env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert (home / "stub-calls").read_text().splitlines() == ["called", "called"]
        assert json.loads(cli("budget", "status").stdout)["unresolved"] == []
        with sqlite3.connect(home / "data/spend.db") as db:
            cost = db.execute("SELECT amount FROM spend WHERE id != ?", (request,)).fetchone()[0]
            assert cost > 0
            assert db.execute("SELECT spent, held, unknown FROM spend_periods").fetchall() == [(5000 + cost, 0, 0)] * 2
            assert db.execute("SELECT evidence FROM spend_reconciliations").fetchall() == [("isolated fixture receipt",)]
        print(json.dumps({"crash_accounting": True, "blocked_dispatches": 1, "stub_calls": 2, "paid_calls": 0}))
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_stuck_connection_exits_and_allows_restart(runtime):
    home, env, cli = runtime
    env["TEST_STUCK_MCP"] = "1"
    port = free_port()
    cli("daemon", "start", "--port", str(port))
    pid = ready(port)["pid"]
    wait_for(lambda: (home / "mcp-started").exists())
    started = time.monotonic()
    stopped = cli("daemon", "stop", ok=False)
    elapsed = time.monotonic() - started
    assert (home / "mcp-cancelled").exists()
    assert elapsed < 35
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    env.pop("TEST_STUCK_MCP")
    cli("daemon", "start")
    assert ready(port)["pid"] != pid
    assert ready(port)["ready"]
    print(json.dumps({"stuck_shutdown_seconds": round(elapsed, 3), "restart_after_watchdog": True}))


@pytest.mark.skipif(sys.platform != "darwin", reason="Requires the macOS user service manager")
def test_launchd_restarts_crashes_but_respects_explicit_stop(runtime, monkeypatch):
    from rune.daemon.service import launchd_config

    home, env, cli = runtime
    domain = f"gui/{os.getuid()}"
    if subprocess.run(["launchctl", "print", domain], capture_output=True, timeout=5).returncode:
        pytest.skip("No logged-in macOS user session")
    port = free_port()
    label = f"com.rune.e2e.{uuid4().hex}"
    target = f"{domain}/{label}"
    with monkeypatch.context() as patch:
        patch.setenv("RUNE_HOME", str(home))
        config = launchd_config(port=port)
    config.update(Label=label, WorkingDirectory=str(home.parent), EnvironmentVariables=env)
    plist = home.parent / "test.plist"
    plist.write_bytes(plistlib.dumps(config))

    def running(previous=None):
        try:
            result = ready(port)
            return result if result["ready"] and result["pid"] != previous else None
        except OSError:
            return None

    installed = False
    try:
        boot = subprocess.run(["launchctl", "bootstrap", domain, str(plist)],
                              capture_output=True, text=True, timeout=10)
        assert boot.returncode == 0, boot.stderr
        installed = True
        first = wait_for(running)
        started = time.monotonic()
        os.kill(first["pid"], signal.SIGKILL)
        second = wait_for(lambda: running(first["pid"]), seconds=20)
        recovery = time.monotonic() - started
        cli("daemon", "stop")
        time.sleep(config["ThrottleInterval"] + .5)
        assert running() is None
        status = subprocess.run(["launchctl", "print", target], capture_output=True, text=True, timeout=5)
        assert status.returncode == 0 and "last exit code = 0" in status.stdout
        print(json.dumps({"launchd_crash_recovery_seconds": round(recovery, 3),
                          "pid_changed": second["pid"] != first["pid"], "explicit_stop_stayed_stopped": True}))
    finally:
        if installed:
            removed = subprocess.run(["launchctl", "bootout", target], capture_output=True, timeout=40)
            assert removed.returncode == 0, removed.stderr
