"""Health handler - GET /health.

Ported from src/api/handlers/health.ts - system status, uptime,
memory usage, and subsystem health checks.
"""

from __future__ import annotations

import os
import platform
import time

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["health"])

VERSION = "0.1.0"
_start_time = time.monotonic()


# Models


class SchedulerStats(BaseModel):
    queued: int = 0
    running: int = 0
    max_concurrency: int = Field(3, alias="maxConcurrency")

    model_config = ConfigDict(populate_by_name=True)


class SubsystemStatus(BaseModel):
    memory: str = "ok"
    proactive: str = "disabled"
    gateway: str = "ok"
    mcp: str = "disabled"
    scheduler: SchedulerStats = Field(default_factory=SchedulerStats)


class MemoryInfo(BaseModel):
    rss_mb: float = Field(0.0, alias="rssMb")
    heap_used_mb: float = Field(0.0, alias="heapUsedMb")

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    subsystems: SubsystemStatus
    memory: MemoryInfo | None = None
    python_version: str = Field("", alias="pythonVersion")
    pid: int = 0

    model_config = ConfigDict(populate_by_name=True)


# Route


def _subsystem_status() -> SubsystemStatus:
    """Report what is actually running.

    Reads the live singletons. Defaults alone would make the endpoint — and the
    overall status derived from it — always say "ok", which is worse than no
    health check at all.
    """
    status = SubsystemStatus()

    try:
        from rune.memory import manager as memory_manager

        status.memory = "ok" if memory_manager._manager is not None else "disabled"
    except Exception:
        status.memory = "error"

    try:
        from rune.daemon.gateway import get_gateway

        status.gateway = "ok" if get_gateway() is not None else "disabled"
    except Exception:
        status.gateway = "error"

    try:
        from rune.daemon.main import _configured_proactive_enabled

        status.proactive = "ok" if _configured_proactive_enabled() else "disabled"
    except Exception:
        status.proactive = "error"

    try:
        from rune.mcp.bridge import get_mcp_status

        # A server appears here once registered, connected or not — report ok
        # only when at least one is actually connected.
        servers = get_mcp_status()["servers"]
        if any(srv.get("connected") for srv in servers):
            status.mcp = "ok"
        elif servers:
            status.mcp = "error"
        else:
            status.mcp = "disabled"
    except Exception:
        status.mcp = "disabled"

    return status


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """System health check.

    This endpoint does **not** require authentication so that
    external monitoring tools can poll it freely.
    """
    uptime = time.monotonic() - _start_time
    subsystems = _subsystem_status()

    # Memory info
    memory_info = MemoryInfo(rssMb=0.0, heapUsedMb=0.0)
    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on Linux, kilobytes on macOS
        rss_kb = rusage.ru_maxrss
        if platform.system() == "Darwin":
            rss_mb = rss_kb / (1024 * 1024)  # bytes -> MB on macOS
        else:
            rss_mb = rss_kb / 1024  # KB -> MB on Linux
        memory_info = MemoryInfo(rssMb=round(rss_mb, 2), heapUsedMb=0.0)
    except Exception:
        pass

    # Overall status
    is_ok = subsystems.memory == "ok" and subsystems.gateway == "ok"
    is_degraded = subsystems.memory == "error" or subsystems.gateway != "ok"
    status = "ok" if is_ok else ("degraded" if is_degraded else "down")

    return HealthResponse(
        status=status,
        version=VERSION,
        uptime=round(uptime, 2),
        subsystems=subsystems,
        memory=memory_info,
        pythonVersion=platform.python_version(),
        pid=os.getpid(),
    )
