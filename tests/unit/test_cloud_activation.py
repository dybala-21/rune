"""Hosted workers cannot start integrations before their host authorizes the boot."""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from rune.cloud import activation


@pytest.fixture
def gate(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir(mode=0o755)
    boot = tmp_path / "boot-id"
    boot.write_text("boot-one\n")
    permit = directory / "activation.json"
    token_file = tmp_path / "api-tokens.json"
    token_file.write_text(json.dumps({"rune_test": {"created_at": 0}}))
    from rune.api import auth

    monkeypatch.setattr(auth, "_TOKEN_FILE", token_file)
    monkeypatch.setattr(activation, "ROOT_UID", os.getuid())
    return permit, boot


def authorize(gate, *, operation="upgrade-one", boot="boot-one"):
    permit, _ = gate
    permit.write_text(json.dumps({"operationId": operation, "bootId": boot}))
    permit.chmod(0o644)


@pytest.mark.asyncio
async def test_standby_has_no_normal_api_and_requires_both_credentials_and_host_permit(gate):
    calls = []
    app = activation.create_activation_app(permit=gate[0], boot_file=gate[1], on_activate=lambda: calls.append(True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
        assert (await client.get("/api/runtime/readiness")).status_code == 401
        client.headers["Authorization"] = "Bearer rune_invalid"
        assert (await client.get("/api/runtime/readiness")).status_code == 403
        client.headers["Authorization"] = "Bearer rune_test"
        response = await client.get("/api/runtime/readiness")
        assert response.json() == {"ready": True, "activationRequired": True, "protocol": 1, "operationId": None}
        for method, path in [("GET", "/api/runs/snapshot"), ("POST", "/api/chat"), ("POST", "/rpc")]:
            assert (await client.request(method, path)).status_code == 503
        assert (await client.post("/api/runtime/activate", json={"operationId": "upgrade-one"})).status_code == 403
        assert calls == []
        authorize(gate)
        assert (await client.post("/api/runtime/activate", json={"operationId": "wrong"})).status_code == 409
        for _ in range(2):
            assert (await client.post("/api/runtime/activate", json={"operationId": "upgrade-one"})).status_code == 202
        authorize(gate, operation="replacement")
        assert (await client.post("/api/runtime/activate", json={"operationId": "replacement"})).status_code == 409
        assert app.state.operation_id == "upgrade-one"


@pytest.mark.asyncio
async def test_bad_activation_input_never_starts_worker(gate):
    app = activation.create_activation_app(permit=gate[0], boot_file=gate[1])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker",
                                headers={"Authorization": "Bearer rune_test"}) as client:
        for content in [b"[", b"[]", b'{}', b'{"operationId":"../unsafe"}']:
            assert (await client.post("/api/runtime/activate", content=content)).status_code == 400
        assert (await client.post("/api/runtime/activate", content=b"x" * 1025)).status_code == 413
        gate[0].write_text("corrupt")
        gate[0].chmod(0o644)
        assert (await client.get("/api/runtime/readiness")).status_code == 503
        assert app.state.operation_id is None


def test_permit_survives_service_restart_but_not_a_different_boot(gate):
    assert activation.read_permit(*gate) is None
    authorize(gate)
    assert activation.read_permit(*gate) == "upgrade-one"
    assert activation.read_permit(*gate) == "upgrade-one"
    gate[1].write_text("boot-two\n")
    assert activation.read_permit(*gate) is None


def test_wrong_owner_writable_paths_and_symlinks_are_rejected(gate, monkeypatch, tmp_path):
    authorize(gate)
    monkeypatch.setattr(activation, "ROOT_UID", os.getuid() + 1)
    with pytest.raises(ValueError, match="directory"):
        activation.read_permit(*gate)
    monkeypatch.setattr(activation, "ROOT_UID", os.getuid())
    gate[0].chmod(0o666)
    with pytest.raises(ValueError, match="permit"):
        activation.read_permit(*gate)
    gate[0].chmod(0o644)
    gate[0].parent.chmod(0o777)
    with pytest.raises(ValueError, match="directory"):
        activation.read_permit(*gate)
    gate[0].parent.chmod(0o755)
    target = tmp_path / "redirect"
    gate[0].rename(target)
    gate[0].symlink_to(target)
    with pytest.raises(OSError):
        activation.read_permit(*gate)


def test_shutdown_while_waiting_does_not_enter_worker(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    marker = tmp_path / "worker-started"
    code = """
import sys
from pathlib import Path
from rune.cloud import activation
activation.read_permit = lambda *args: None
activation.wait_for_activation(host='127.0.0.1', port=int(sys.argv[1]))
Path(sys.argv[2]).write_text('started')
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(port), str(marker)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env={**os.environ, "RUNE_HOME": str(tmp_path)})
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("Standby listener did not start")
        process.terminate()
        process.wait(timeout=8)
        assert not marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()


def test_valid_same_boot_permit_does_not_open_another_listener(monkeypatch):
    import uvicorn

    monkeypatch.setattr(activation, "read_permit", lambda: "same-boot")

    def forbidden(*args, **kwargs):
        pytest.fail("A service restart should reuse its host permit")

    monkeypatch.setattr(uvicorn, "Server", forbidden)
    activation.wait_for_activation()


def test_activation_continues_only_after_authenticated_host_authorization(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = tmp_path / "run"
    directory.mkdir(mode=0o755)
    permit = directory / "activation.json"
    boot = tmp_path / "boot-id"
    boot.write_text("boot-one\n")
    (tmp_path / "api-tokens.json").write_text(json.dumps({"rune_test": {"created_at": 0}}))
    marker = tmp_path / "worker-started"
    code = """
import os, sys
from pathlib import Path
from rune.cloud import activation
activation.ROOT_UID = os.getuid()
read = activation.read_permit
activation.read_permit = lambda *args: read(Path(sys.argv[2]), Path(sys.argv[3]))
activation.wait_for_activation(host='127.0.0.1', port=int(sys.argv[1]))
assert 'rune.daemon.main' not in sys.modules
Path(sys.argv[4]).write_text('started')
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(port), str(permit), str(boot), str(marker)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env={**os.environ, "RUNE_HOME": str(tmp_path)})
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=1,
                          headers={"Authorization": "Bearer rune_test"}) as client:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(process.stderr.read().decode())
                try:
                    response = client.get("/api/runtime/readiness")
                    assert response.status_code == 200
                    break
                except httpx.TransportError:
                    time.sleep(0.05)
            else:
                pytest.fail("Standby listener did not start")
            assert not marker.exists()
            assert client.post("/api/runtime/activate", json={"operationId": "upgrade-one"}).status_code == 403
            authorize((permit, boot))
            assert client.post("/api/runtime/activate", json={"operationId": "upgrade-one"}).status_code == 202
        process.wait(timeout=8)
        assert process.returncode == 0, process.stderr.read().decode()
        assert marker.read_text() == "started"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
