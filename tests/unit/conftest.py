"""Keep unit tests off the developer's real Rune home, credentials and the network."""

import ipaddress
import os
import re
import socket

import pytest

# Importing rune loads ~/.rune/.env into the process before any fixture runs, so credentials are
# removed per test by name; a test that needs one sets its own fake value.
_CREDENTIAL = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_ACCESS_KEY(_ID)?|_CREDENTIALS?)$|^AWS_|^GOOGLE_APPLICATION")
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _local(address) -> bool:
    if isinstance(address, (str, bytes)):  # Unix domain socket
        return True
    host = address[0] if isinstance(address, tuple) and address else ""
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guard(connect):
    def guarded(self, address):
        if not _local(address):
            raise OSError(f"Unit tests may not reach the network ({address!r}); mock the client instead.")
        return connect(self, address)
    return guarded


@pytest.fixture(autouse=True)
def isolated_rune_home(tmp_path_factory, monkeypatch):
    # The real ~/.rune holds config, credentials and memory; unit tests must never touch it.
    home = tmp_path_factory.mktemp("rune-home")
    monkeypatch.setenv("RUNE_HOME", str(home))
    monkeypatch.setenv("RUNE_BROKER_HOME", str(home / "broker"))
    monkeypatch.setenv("RUNE_CLOUD_HOME", str(home / "cloud"))
    for name in [name for name in os.environ if _CREDENTIAL.search(name)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _guard(_real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(_real_connect_ex))
    from rune.agent import decision_router
    from rune.config.loader import reset_config

    monkeypatch.setattr(decision_router, "_cooldown", None)
    monkeypatch.setattr(decision_router, "_rejected_key", None)
    monkeypatch.setattr(decision_router, "_verified_key", None)
    reset_config()
