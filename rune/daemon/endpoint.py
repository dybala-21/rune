"""Share the daemon address between CLI, services, and the desktop shell."""

import ipaddress
import json
import os
import tempfile
from dataclasses import dataclass

from rune.utils.paths import rune_home


@dataclass(frozen=True)
class Endpoint:
    host: str = "127.0.0.1"
    port: int = 18789

    def __post_init__(self):
        if not isinstance(self.host, str):
            raise ValueError("Daemon host must be an IP address or localhost")
        if self.host != "localhost":
            ipaddress.ip_address(self.host)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("Daemon port must be between 1 and 65535")

    @property
    def connect_host(self) -> str:
        return {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)


def saved_endpoint() -> Endpoint:
    try:
        raw = json.loads((rune_home() / "daemon.json").read_text())
    except FileNotFoundError:
        return Endpoint()
    return Endpoint(raw["host"], raw["port"])


def resolve_endpoint(host=None, port=None) -> Endpoint:
    if host is not None and port is not None:
        return Endpoint(host, port)
    saved = saved_endpoint()
    value = port if port is not None else os.environ.get("RUNE_API_PORT", saved.port)
    return Endpoint(host if host is not None else os.environ.get("RUNE_API_HOST", saved.host), int(value))


def save_endpoint(endpoint: Endpoint) -> None:
    path = rune_home() / "daemon.json"
    fd, temporary = tempfile.mkstemp(prefix=".daemon-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"host": endpoint.host, "port": endpoint.port}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
