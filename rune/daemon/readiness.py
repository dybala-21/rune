"""Check readiness and confirm the response belongs to the expected daemon."""

import json
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def is_ready(host: str, port: int, pid: int) -> bool:
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host:
        host = f"[{host}]"
    try:
        with build_opener(ProxyHandler({})).open(f"http://{host}:{port}/ready", timeout=0.3) as response:
            data = json.load(response)
        return data.get("ready") is True and data.get("pid") == pid
    except (OSError, ValueError, URLError):
        return False
