"""Test daemon shutdown and crash recovery with an isolated, waiting model stub."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
python = sys.executable
child = """import asyncio, os, runpy, sys
from pathlib import Path
from rune.agent import timing
async def denied(*args, **kwargs):
    Path(os.environ['RUNE_HOME'], 'unexpected-model-call').touch()
    raise AssertionError('Paid calls are disabled in this process test')
timing.timed_completion = denied
from rune.agent import loop
class HeldLoop:
    def __init__(self, **kwargs):
        self._last_answer_text = ''
        self._last_goal_type = 'chat'
    def set_approval_callback(self, cb): pass
    def set_ask_user_callback(self, cb): pass
    def on(self, *args): pass
    async def cancel(self): pass
    async def run(self, *args, **kwargs):
        Path(os.environ['RUNE_HOME'], 'run-started').touch()
        try:
            await asyncio.Event().wait()
        finally:
            Path(os.environ['RUNE_HOME'], 'run-cleaned').touch()
loop.NativeAgentLoop = HeldLoop
from rune.daemon.main import RuneDaemon
async def slow_mcp(self):
    await asyncio.Event().wait()
RuneDaemon._initialize_mcp_bridge = slow_mcp
sys.argv = ['rune', 'daemon', 'run', '--port', os.environ['TEST_PORT']]
runpy.run_module('rune.cli.main', run_name='__main__')
"""


def test_daemon_shutdown_and_crash_recovery():
    with tempfile.TemporaryDirectory(prefix="rune-uptime-", dir="/tmp") as directory:
        root = Path(directory)
        home = root / "state"
        home.mkdir()
        (home / "config.yaml").write_text(
            "proactive:\n  enabled: false\nskills:\n  gatedLearning: false\n"
        )
        runner = root / "runner.py"
        runner.write_text(child)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {
            key: os.environ[key]
            for key in ("HOME", "PATH", "USER", "TMPDIR", "LANG")
            if key in os.environ
        }
        env.update(
            RUNE_HOME=str(home),
            RUNE_WORKSPACE=str(root / "work"),
            RUNE_WEB_STATIC_DIR=str(repo / "web/dist"),
            PYTHONPATH=str(repo),
            TEST_PORT=str(port),
            RUNE_LOG_FILE=str(root / "runtime.log"),
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(path, data=None):
            raw = json.dumps(data).encode() if data is not None else None
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", raw, {"Content-Type": "application/json"}
            )
            with opener.open(req, timeout=1) as response:
                return json.load(response)

        def wait_for(check, seconds=20):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    result = check()
                    if result:
                        return result
                except (OSError, ValueError):
                    pass
                time.sleep(0.05)
            raise AssertionError("timed out")

        processes = []
        observations = []

        def start():
            t = time.monotonic()
            proc = subprocess.Popen(
                [python, str(runner)],
                cwd=root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(proc)
            ready = wait_for(lambda: request("/ready"))
            assert ready["pid"] == proc.pid and ready["ready"]
            assert ready["connections"]["mcp"] == "connecting"
            observations.append({"startup_seconds": round(time.monotonic() - t, 3)})
            return proc

        try:
            first = start()
            request(
                "/api/message",
                {"text": "Hold this isolated lifecycle test", "sessionId": "lifecycle-test"},
            )
            wait_for(lambda: (home / "run-started").exists())
            start_time = time.monotonic()
            first.send_signal(signal.SIGTERM)
            assert first.wait(timeout=30) == 0
            observations.append(
                {
                    "shutdown_seconds": round(time.monotonic() - start_time, 3),
                    "active_run_cleaned": (home / "run-cleaned").exists(),
                    "pid_removed": not (home / "data/rune.pid").exists(),
                    "socket_removed": not (home / "data/rune.sock").exists(),
                }
            )
            assert (home / "run-cleaned").exists()
            assert not (home / "data/rune.pid").exists() and not (home / "data/rune.sock").exists()
            second = start()
            snap = request("/api/runs/snapshot?sessionId=lifecycle-test")["run"]
            assert snap["status"] not in ("running", "queued")
            (home / "run-started").unlink()
            request("/api/message", {"text": "Hold until crash", "sessionId": "crash-test"})
            wait_for(lambda: (home / "run-started").exists())
            second.kill()
            second.wait(timeout=5)
            third = start()
            snap = request("/api/runs/snapshot?sessionId=crash-test")["run"]
            assert snap["status"] == "interrupted", snap["status"]
            observations.append(
                {
                    "crash_run_status": snap["status"],
                    "ready_after_crash": request("/ready")["ready"],
                }
            )
            stopped = subprocess.run([python, "-m", "rune.cli.main", "daemon", "stop"],
                                     cwd=root, env=env, capture_output=True, text=True, timeout=35)
            assert stopped.returncode == 0, stopped.stderr
            assert third.wait(timeout=30) == 0
            assert not (home / "unexpected-model-call").exists()
            print(json.dumps({"passed": True, "model_calls": 0, "observations": observations}))
        except BaseException:
            print(
                (root / "runtime.log").read_text()[-12000:]
                if (root / "runtime.log").exists()
                else "no runtime log"
            )
            raise
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
