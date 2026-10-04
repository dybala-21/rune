"""Generated checks have the same command authority as ordinary tools."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from dataclasses import dataclass
from uuid import uuid4

from rune.safety.approval_context import approval_required
from rune.safety.execution_environment import execution_config, execution_workspace
from rune.safety.process_env import child_environment
from rune.safety.resource_locks import ResourceBusy, command_access
from rune.utils.logger import get_logger

log = get_logger(__name__)


def verification_blocker(command: str, cwd: str) -> str | None:
    from rune.capabilities.bash import _configured_deny_by_default, _configured_rollout_mode
    from rune.safety.execution_policy import ExecutionPolicyConfig, decide_bash_execution
    from rune.safety.guardian import get_guardian

    enabled, allowed = _configured_deny_by_default()
    decision = decide_bash_execution(
        command, get_guardian().validate(command, cwd=cwd),
        ExecutionPolicyConfig(rollout_mode=_configured_rollout_mode(), sandbox_enabled=False,
                              deny_by_default_enabled=enabled, allowed_executables=allowed),
        has_sandbox_support=False, interactive_approval=False,
    )
    return None if decision.decision == "allow" else decision.reason


@dataclass(frozen=True)
class CheckResult:
    code: int | None = None
    stdout: bytes = b""
    error: str = ""


async def run_check(command: str, cwd: str, timeout: float, *, limit: int = 64 * 1024 * 1024) -> CheckResult:
    """A missing verdict is inconclusive, including blocked or truncated checks."""
    try:
        with approval_required():
            if reason := verification_blocker(command, cwd):
                return CheckResult(error=reason)
            with command_access(command, cwd):
                return await _run_check(command, cwd, timeout, limit)
    except (OSError, ResourceBusy, ValueError) as exc:
        return CheckResult(error=str(exc))


async def _run_check(command: str, cwd: str, timeout: float, limit: int) -> CheckResult:
    config = execution_config()
    env = child_environment()
    container = None
    if config.backend == "container":
        from rune.safety.container_exec import container_argv

        if shutil.which("docker") is None:
            return CheckResult(error="Docker is unavailable; verification was not run.")
        container = f"rune-check-{uuid4().hex}"
        argv = container_argv(command, cwd, image=config.image, name=container,
                              network=config.allow_network, workspace=execution_workspace(cwd))
        argv.remove("--rm")  # Cleanup below must confirm removal, including after cancellation.
        env.pop("DOCKER_HOST", None)
        env.pop("DOCKER_CONTEXT", None)
    else:
        argv = ["sh", "-c", command]
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    overflow = False

    async def drain(stream):
        nonlocal overflow
        output = bytearray()
        while chunk := await stream.read(65536):
            remaining = max(0, limit - len(output))
            output.extend(chunk[:remaining])
            overflow |= len(chunk) > remaining
        return bytes(output)

    stdout = asyncio.create_task(drain(proc.stdout))
    stderr = asyncio.create_task(drain(proc.stderr))
    code = None
    error = ""
    try:
        async with asyncio.timeout(timeout):
            # A background child can keep pipes open after the shell exits.
            while proc.returncode is None:
                await asyncio.sleep(0.02)
            code = proc.returncode
    except TimeoutError:
        error = "Verification timed out; its result is unknown."
    finally:
        if container:
            from rune.safety.container_exec import remove_container
            if not await remove_container(container, env):
                code, error = None, "Could not confirm verification container cleanup."
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            log.debug("verification_process_exited", pid=proc.pid)
        try:
            async with asyncio.timeout(3):
                await asyncio.gather(stdout, stderr, proc.wait())
        except TimeoutError:
            code, error = None, "Could not drain verification output."
            stdout.cancel()
            stderr.cancel()
            await asyncio.gather(stdout, stderr, return_exceptions=True)
    if overflow:
        return CheckResult(error="Verification output exceeded the capture limit.")
    out = stdout.result() if stdout.done() and not stdout.cancelled() else b""
    err = stderr.result() if stderr.done() and not stderr.cancelled() else b""
    return CheckResult(code, out, error or err.decode("utf-8", "replace"))
