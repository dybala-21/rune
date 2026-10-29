"""Send local control requests without signalling an unverified PID."""

import json
import socket


def request(path: str, command: str, **params) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        connection.connect(path)
        connection.sendall(json.dumps({"command": command, "params": params}).encode() + b"\n")
        with connection.makefile("rb") as stream:
            line = stream.readline(65537)
        if len(line) > 65536 or not line.endswith(b"\n"):
            raise RuntimeError("Rune returned an incomplete control response.")
    response = json.loads(line)
    if not isinstance(response, dict) or not response.get("success"):
        raise RuntimeError("Rune did not accept the control request.")
    if params.get("expected_pid") is not None and response.get("pid") != params["expected_pid"]:
        raise RuntimeError("The Rune instance changed; retry the request.")
    return response
