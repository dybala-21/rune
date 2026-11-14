"""Hold a hosted worker before startup until the host authorizes this boot."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

PROTOCOL = 1
PERMIT = Path("/run/rune-worker/activation.json")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
ROOT_UID = 0


def valid_operation(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def read_permit(path: Path = PERMIT, boot_file: Path = BOOT_ID) -> str | None:
    """Accept only a root-controlled permit for the current boot."""
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    try:
        parent = os.fstat(directory)
        if parent.st_uid != ROOT_UID or parent.st_mode & 0o022:
            raise ValueError("Worker activation directory is not controlled by root")
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != ROOT_UID
                    or metadata.st_mode & 0o022 or metadata.st_size > 4096):
                raise ValueError("Worker activation permit is not controlled by root")
            permit = json.loads(stream.read(4097))
        if not isinstance(permit, dict) or not valid_operation(permit.get("operationId")):
            raise ValueError("Worker activation permit has no valid operation")
        boot = boot_file.read_text().strip()
        if not boot or permit.get("bootId") != boot:
            return None
        return permit["operationId"]
    finally:
        os.close(directory)


def create_activation_app(*, permit: Path = PERMIT, boot_file: Path = BOOT_ID,
                          on_activate: Callable[[], None] | None = None) -> FastAPI:
    """Serve readiness and activation before the agent or its integrations start."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.operation_id = None

    def authenticate(request: Request) -> None:
        from rune.api.auth import verify_token

        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Worker authentication required")
        if not verify_token(token):
            raise HTTPException(403, "Invalid worker token")

    def authorized_operation() -> str | None:
        try:
            return read_permit(permit, boot_file)
        except (OSError, ValueError) as error:
            raise HTTPException(503, "Worker activation permit is invalid") from error

    @app.get("/api/runtime/readiness")
    async def readiness(request: Request):
        authenticate(request)
        return {"ready": True, "activationRequired": True, "protocol": PROTOCOL,
                "operationId": authorized_operation()}

    @app.post("/api/runtime/activate")
    async def activate(request: Request):
        authenticate(request)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1024:
                raise HTTPException(413, "Activation request is too large")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError) as error:
            raise HTTPException(400, "Invalid activation request") from error
        operation = payload.get("operationId") if isinstance(payload, dict) else None
        if not valid_operation(operation):
            raise HTTPException(400, "Invalid activation operation")
        allowed = authorized_operation()
        if allowed is None:
            raise HTTPException(403, "The host has not authorized this boot")
        if operation != allowed or app.state.operation_id not in (None, operation):
            raise HTTPException(409, "Activation operation does not match")
        app.state.operation_id = operation
        if on_activate is not None:
            on_activate()
        return JSONResponse({"accepted": True, "operationId": operation}, status_code=202)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def unavailable(request: Request):
        authenticate(request)
        return JSONResponse({"detail": "Worker is waiting for host activation"}, status_code=503,
                            headers={"Retry-After": "2"})

    return app


def wait_for_activation(*, host: str = "0.0.0.0", port: int = 18789) -> None:
    """Return after authorization; a same-boot service restart retains its permit."""
    if read_permit() is not None:
        return
    import uvicorn

    def activated() -> None:
        server.should_exit = True

    app = create_activation_app(on_activate=activated)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", log_config=None,
                                          access_log=False, timeout_graceful_shutdown=5))
    server.run()
    # SIGTERM or a failed listener must never fall through to normal worker startup.
    if app.state.operation_id is None or read_permit() != app.state.operation_id:
        raise SystemExit("Worker stopped before host activation")
