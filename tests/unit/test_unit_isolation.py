"""The unit suite never reaches the network or the developer's real credentials."""

import os
import socket

import pytest


def test_outbound_connections_are_refused():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(OSError, match="may not reach the network"):
        sock.connect(("93.184.216.34", 443))


def test_loopback_stays_available():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.socket() as client:
            client.connect(server.getsockname())


def test_no_real_credentials_or_home_are_visible():
    assert not [name for name in os.environ if name.endswith(("_API_KEY", "_TOKEN", "_SECRET"))]
    assert os.path.expanduser("~/.rune") != os.environ["RUNE_HOME"]
