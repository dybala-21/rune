"""Bash capability for RUNE.

Ported from src/capabilities/bash.ts - subprocess execution with
Guardian validation, sandbox support, and managed service mode.

Managed-service mode implements a full lifecycle:
  spawn -> readiness probe -> smoke verification -> teardown
mirroring the TS ``executeManagedService`` implementation.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import re
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.config.defaults import (
    DEFAULT_BASH_TIMEOUT_MS,
    DEFAULT_OUTPUT_BUFFER_LIMIT,
)
from rune.safety.execution_policy import (
    DEFAULT_ALLOWED_EXECUTABLES,
    ExecutionPolicyConfig,
    decide_bash_execution,
)
from rune.safety.guardian import get_guardian
from rune.types import CapabilityResult, Domain, RiskLevel
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

_TAIL_LINES = 20
_READINESS_MARKERS = re.compile(
    r"\blistening on\b|\bgateway listening\b|\badmin listening\b"
    r"|\bserver started\b|\bready to accept\b",
    re.IGNORECASE,
)

# Managed-process tracking


@dataclass
class ManagedProcess:
    """Bookkeeping for a background (managed-service) process."""
    service_id: str
    pid: int
    command: str
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.monotonic)
    cwd: str = ""


# Module-level dict - survives across calls within the same interpreter.
_managed_processes: dict[str, ManagedProcess] = {}
_cleanup_registered = False


def _kill_process_group(pid: int, sig: int = signal.SIGTERM) -> None:
    """Send *sig* to the entire process group rooted at *pid*."""
    if not hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, sig)
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), sig)


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _cleanup_managed_processes() -> None:
    """Terminate **all** tracked managed processes (best-effort).

    Called at interpreter exit and can also be invoked explicitly.
    """
    for sid, mp in list(_managed_processes.items()):
        log.info("cleanup_managed_process", service_id=sid, pid=mp.pid)
        _kill_process_group(mp.pid, signal.SIGTERM)
        # Give it a short grace period, then escalate.
        if _is_process_alive(mp.pid):
            _kill_process_group(mp.pid, signal.SIGKILL)
    _managed_processes.clear()


def _ensure_cleanup_handler() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True
    atexit.register(_cleanup_managed_processes)


# Helpers

def _tail_lines(text: str, n: int = _TAIL_LINES) -> str:
    lines = text.split("\n")
    return "\n".join(lines[-n:])


def _has_startup_markers(text: str) -> bool:
    return bool(_READINESS_MARKERS.search(text))


# Parameters


class BashParams(BaseModel):
    command: str = Field(description="Shell command to execute")
    cwd: str | None = Field(default=None, description="Working directory")
    timeout: int = Field(default=DEFAULT_BASH_TIMEOUT_MS, description="Timeout in ms")
    env: dict[str, str] | None = Field(default=None, description="Extra env vars")
    mode: str = Field(
        default="oneshot",
        description=(
            "oneshot: single command execution, "
            "managed_service: service start/ready/probe/cleanup lifecycle"
        ),
    )
    # managed_service parameters
    readiness_command: str | None = Field(
        default=None, description="Command to probe readiness"
    )
    readiness_timeout: int = Field(
        default=15_000, description="Max wait for readiness (ms)"
    )
    readiness_interval: int = Field(
        default=1_000, description="Readiness probe retry interval (ms)"
    )
    smoke_command: str | None = Field(
        default=None, description="Smoke verification command after readiness"
    )
    smoke_timeout: int = Field(
        default=10_000, description="Smoke command timeout (ms)"
    )
    teardown_command: str | None = Field(
        default=None, description="Teardown / cleanup command"
    )
    teardown_timeout: int = Field(
        default=10_000, description="Teardown command timeout (ms)"
    )
    log_file: str | None = Field(
        default=None, description="File to append service stdout/stderr"
    )


# Service status / stop params


class ServiceStatusParams(BaseModel):
    service_id: str | None = Field(
        default=None,
        description="Service ID to query; omit for all services",
    )


class ServiceStopParams(BaseModel):
    service_id: str = Field(description="Service ID to stop")


# Oneshot execution (unchanged logic, extracted for clarity)


async def _execute_oneshot(params: BashParams) -> CapabilityResult:
    """Run a single command and wait for completion."""
    cwd = params.cwd or os.getcwd()
    env = {**os.environ, **(params.env or {})}
    timeout_sec = params.timeout / 1000.0

    try:
        proc = await asyncio.create_subprocess_shell(
            params.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            if proc.pid and hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:
                proc.kill()
            await proc.wait()

            return CapabilityResult(
                success=False,
                error=f"Command timed out after {params.timeout}ms",
                metadata={"timeout": True},
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if len(stdout) > DEFAULT_OUTPUT_BUFFER_LIMIT:
            stdout = stdout[:DEFAULT_OUTPUT_BUFFER_LIMIT] + "\n... (truncated)"
        if len(stderr) > DEFAULT_OUTPUT_BUFFER_LIMIT:
            stderr = stderr[:DEFAULT_OUTPUT_BUFFER_LIMIT] + "\n... (truncated)"

        success = proc.returncode == 0
        output = stdout
        if stderr and not success:
            output += f"\nSTDERR:\n{stderr}"

        return CapabilityResult(
            success=success,
            output=output,
            error=stderr if not success else None,
            metadata={"exit_code": proc.returncode, "cwd": cwd},
        )

    except FileNotFoundError:
        return CapabilityResult(success=False, error="Shell not found")
    except Exception as exc:
        return CapabilityResult(success=False, error=f"Execution failed: {exc}")


# Managed-service lifecycle


async def _run_probe(
    command: str,
    *,
    cwd: str,
    env: dict[str, str],
    timeout_ms: int,
) -> tuple[int | None, str, str]:
    """Run a short-lived probe command; returns (exit_code, stdout, stderr)."""
    timeout_sec = max(timeout_ms / 1000.0, 0.5)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_sec,
        )
        return (
            proc.returncode,
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
        )
    except TimeoutError:
        return (None, "", "probe timed out")
    except Exception as exc:
        return (None, "", str(exc))


async def _execute_managed_service(params: BashParams) -> CapabilityResult:
    """Spawn a background service and run the readiness/smoke/teardown lifecycle.

    Closely follows the TS ``executeManagedService`` implementation.
    """
    _ensure_cleanup_handler()

    cwd = params.cwd or os.getcwd()
    env = {**os.environ, **(params.env or {})}
    service_id = uuid.uuid4().hex[:12]

    # Phase 1 - spawn
    try:
        proc = await asyncio.create_subprocess_shell(
            params.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"[E_SERVICE_NOT_READY] Failed to spawn managed service: {exc}",
            metadata={"reasonCode": "E_SERVICE_NOT_READY", "stage": "spawn_managed"},
        )

    if proc.pid is None:
        return CapabilityResult(
            success=False,
            error="[E_SERVICE_NOT_READY] Failed to spawn managed service process",
            metadata={"reasonCode": "E_SERVICE_NOT_READY", "stage": "spawn_managed"},
        )

    pid = proc.pid
    mp = ManagedProcess(
        service_id=service_id,
        pid=pid,
        command=params.command,
        proc=proc,
        cwd=cwd,
    )
    _managed_processes[service_id] = mp
    log.info("managed_service_spawned", service_id=service_id, pid=pid)

    # Collect stdout/stderr in background
    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    child_closed = False
    exit_code: int | None = None
    log_fh = None

    if params.log_file:
        with contextlib.suppress(OSError):
            log_fh = open(params.log_file, "a")

    async def _drain_stream(
        stream: asyncio.StreamReader | None, buf: list[str]
    ) -> None:
        nonlocal child_closed, exit_code
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                buf.append(text)
                if log_fh:
                    try:
                        log_fh.write(text)
                        log_fh.flush()
                    except OSError:
                        pass
        except Exception:
            pass

    drain_out = asyncio.create_task(_drain_stream(proc.stdout, stdout_buf))
    drain_err = asyncio.create_task(_drain_stream(proc.stderr, stderr_buf))

    # Lifecycle metadata
    lifecycle: dict[str, Any] = {
        "mode": "managed_service",
        "pid": pid,
        "service_id": service_id,
        "readiness": {"attempted": False, "success": False},
        "smoke": {"attempted": False, "success": False},
        "teardown": {"attempted": False, "success": False},
    }

    readiness_ready = False
    readiness_error = ""
    stage = "wait_ready"

    try:
        # Phase 2 - readiness
        lifecycle["readiness"]["attempted"] = True
        deadline = time.monotonic() + max(params.readiness_timeout / 1000.0, 0.001)

        while time.monotonic() < deadline:
            # Check if child already exited
            if proc.returncode is not None:
                readiness_error = f"service exited before ready (code={proc.returncode})"
                break

            readiness_cmd = (params.readiness_command or "").strip()
            if readiness_cmd:
                rc, p_out, p_err = await _run_probe(
                    readiness_cmd,
                    cwd=cwd,
                    env=env,
                    timeout_ms=min(4000, max(500, params.readiness_interval)),
                )
                if rc == 0:
                    readiness_ready = True
                    break
                probe_error = (p_err.strip() or p_out.strip() or f"exit={rc}")
                readiness_error = probe_error
                if rc == 127 and "command not found" in probe_error.lower():
                    break
            else:
                combined = "".join(stdout_buf) + "\n" + "".join(stderr_buf)
                if _has_startup_markers(combined):
                    readiness_ready = True
                    break

            await asyncio.sleep(params.readiness_interval / 1000.0)

        if not readiness_ready:
            reason_code = (
                "E_RUNTIME_PROBE_UNAVAILABLE"
                if "command not found" in readiness_error
                else "E_SERVICE_NOT_READY"
            )
            lifecycle["readiness"]["reasonCode"] = reason_code
            if readiness_error:
                lifecycle["readiness"]["detail"] = readiness_error

            tail_text = _tail_lines(
                ("".join(stdout_buf) or "".join(stderr_buf)).strip()
            )
            return CapabilityResult(
                success=False,
                output=tail_text,
                error=(
                    f"[{reason_code}] managed service was not ready within "
                    f"{params.readiness_timeout}ms"
                    + (f": {readiness_error}" if readiness_error else "")
                ),
                metadata={**lifecycle, "reasonCode": reason_code, "stage": "wait_ready"},
            )

        lifecycle["readiness"]["success"] = True

        # Phase 3 - smoke verification
        smoke_cmd = (params.smoke_command or "").strip()
        if smoke_cmd:
            stage = "smoke_verify"
            lifecycle["smoke"]["attempted"] = True
            lifecycle["smoke"]["command"] = smoke_cmd

            rc, s_out, s_err = await _run_probe(
                smoke_cmd, cwd=cwd, env=env, timeout_ms=params.smoke_timeout,
            )
            if rc is None or rc != 0:
                detail = s_err.strip() or s_out.strip() or f"exit={rc}"
                lifecycle["smoke"]["reasonCode"] = "E_SERVICE_NOT_READY"
                lifecycle["smoke"]["detail"] = detail
                return CapabilityResult(
                    success=False,
                    output=_tail_lines((s_out or s_err).strip()),
                    error=f"[E_SERVICE_NOT_READY] smoke verification failed: {detail}",
                    metadata={**lifecycle, "reasonCode": "E_SERVICE_NOT_READY", "stage": "smoke_verify"},
                )
            lifecycle["smoke"]["success"] = True

        # Success
        phases = [
            "- spawn_managed: ok",
            "- wait_ready: ok",
            f"- smoke_verify: {'ok' if smoke_cmd else 'skipped'}",
        ]
        return CapabilityResult(
            success=True,
            output="Managed service lifecycle completed\n" + "\n".join(phases),
            metadata={
                **lifecycle,
                "service_id": service_id,
                "pid": pid,
            },
        )

    except Exception as exc:
        reason_code = "E_SERVICE_NOT_READY"
        lifecycle.setdefault(stage, {})
        return CapabilityResult(
            success=False,
            output=_tail_lines(("".join(stdout_buf) or "".join(stderr_buf)).strip()),
            error=f"[{reason_code}] managed service lifecycle aborted at {stage}: {exc}",
            metadata={**lifecycle, "reasonCode": reason_code, "stage": stage},
        )
    finally:
        # Phase 4 - teardown
        stage = "teardown"
        lifecycle["teardown"]["attempted"] = True
        cleanup_incomplete = False

        teardown_cmd = (params.teardown_command or "").strip()
        if teardown_cmd:
            rc, t_out, t_err = await _run_probe(
                teardown_cmd, cwd=cwd, env=env, timeout_ms=params.teardown_timeout,
            )
            if rc is None or rc != 0:
                cleanup_incomplete = True
                lifecycle["teardown"]["reasonCode"] = "E_CLEANUP_INCOMPLETE"
                lifecycle["teardown"]["detail"] = (
                    t_err.strip() or t_out.strip() or f"exit={rc}"
                )

        # Kill the service process group
        _kill_process_group(pid, signal.SIGTERM)
        await asyncio.sleep(0.15)
        if _is_process_alive(pid):
            cleanup_incomplete = True
            _kill_process_group(pid, signal.SIGKILL)

        if not cleanup_incomplete:
            lifecycle["teardown"]["success"] = True

        # Remove from tracking
        _managed_processes.pop(service_id, None)

        # Clean up drain tasks
        drain_out.cancel()
        drain_err.cancel()
        if log_fh:
            with contextlib.suppress(OSError):
                log_fh.close()

        if cleanup_incomplete:
            log.warning(
                "managed_service_cleanup_incomplete",
                service_id=service_id,
                pid=pid,
            )


# Service status / stop


async def managed_service_status(params: ServiceStatusParams) -> CapabilityResult:
    """Return status of tracked managed services."""
    if params.service_id:
        mp = _managed_processes.get(params.service_id)
        if mp is None:
            return CapabilityResult(
                success=False,
                error=f"No managed service with id '{params.service_id}'",
            )
        alive = _is_process_alive(mp.pid)
        return CapabilityResult(
            success=True,
            output=f"service_id={mp.service_id} pid={mp.pid} alive={alive} cmd={mp.command!r}",
            metadata={
                "service_id": mp.service_id,
                "pid": mp.pid,
                "alive": alive,
                "command": mp.command,
                "uptime_sec": round(time.monotonic() - mp.started_at, 1),
            },
        )

    # All services
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for sid, mp in _managed_processes.items():
        alive = _is_process_alive(mp.pid)
        entries.append({
            "service_id": sid,
            "pid": mp.pid,
            "alive": alive,
            "command": mp.command,
            "uptime_sec": round(time.monotonic() - mp.started_at, 1),
        })
        lines.append(f"  {sid}  pid={mp.pid}  alive={alive}  {mp.command!r}")

    return CapabilityResult(
        success=True,
        output=f"Managed services ({len(entries)}):\n" + "\n".join(lines) if lines else "No managed services running.",
        metadata={"services": entries},
    )


async def managed_service_stop(params: ServiceStopParams) -> CapabilityResult:
    """Stop a specific managed service by service_id."""
    mp = _managed_processes.pop(params.service_id, None)
    if mp is None:
        return CapabilityResult(
            success=False,
            error=f"No managed service with id '{params.service_id}'",
        )

    _kill_process_group(mp.pid, signal.SIGTERM)
    await asyncio.sleep(0.3)
    if _is_process_alive(mp.pid):
        _kill_process_group(mp.pid, signal.SIGKILL)
        await asyncio.sleep(0.1)

    alive = _is_process_alive(mp.pid)
    return CapabilityResult(
        success=not alive,
        output=f"Service {params.service_id} (pid={mp.pid}) {'still alive (SIGKILL sent)' if alive else 'stopped'}.",
        metadata={"service_id": params.service_id, "pid": mp.pid, "alive": alive},
    )


# Main entry point


async def bash_execute(params: BashParams) -> CapabilityResult:
    """Execute a shell command with safety validation."""
    guardian = get_guardian()

    # Validate command through Guardian
    validation = guardian.validate(params.command)

    # Apply execution policy
    policy_config = ExecutionPolicyConfig(
        rollout_mode="balanced",
        allowed_executables=list(DEFAULT_ALLOWED_EXECUTABLES),
    )
    decision = decide_bash_execution(
        params.command, validation, policy_config,
        has_sandbox_support=False,
        interactive_approval=True,
    )

    if decision.decision == "deny":
        return CapabilityResult(
            success=False,
            error=f"Command blocked: {decision.reason}",
        )

    if decision.decision == "ask":
        return CapabilityResult(
            success=False,
            error=f"Command requires approval: {decision.reason}",
            metadata={"requires_approval": True, "reason": decision.reason},
        )

    if params.mode == "managed_service":
        return await _execute_managed_service(params)

    return await _execute_oneshot(params)


# Registration


def register_bash_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(CapabilityDefinition(
        name="bash_execute",
        description="Execute a shell command",
        domain=Domain.PROCESS,
        risk_level=RiskLevel.HIGH,
        group="runtime",
        parameters_model=BashParams,
        execute=bash_execute,
    ))
    registry.register(CapabilityDefinition(
        name="managed_service_status",
        description="Check status of managed background services",
        domain=Domain.PROCESS,
        risk_level=RiskLevel.LOW,
        group="service",
        parameters_model=ServiceStatusParams,
        execute=managed_service_status,
    ))
    registry.register(CapabilityDefinition(
        name="managed_service_stop",
        description="Stop a managed background service",
        domain=Domain.PROCESS,
        risk_level=RiskLevel.MEDIUM,
        group="service",
        parameters_model=ServiceStopParams,
        execute=managed_service_stop,
    ))
