"""Opt-in command isolation. Missing runtimes never fall back to the host."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

from rune.capabilities.command_output import capture_output, stop_capture
from rune.safety.process_env import child_environment
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


def container_argv(command: str, cwd: str, *, image: str, name: str,
                   network: bool = False, env_names: tuple[str, ...] = (), workspace: str | None = None) -> list[str]:
    workdir = Path(cwd).resolve(strict=True)
    root = Path(workspace or cwd).resolve(strict=True)
    if not workdir.is_relative_to(root):
        raise ValueError("The command directory is outside the execution workspace")
    from rune.cloud.store import cloud_home
    from rune.connectors.store import broker_home
    from rune.utils.paths import rune_home

    private = (Path.home().resolve(), rune_home().resolve(), broker_home().resolve(), cloud_home().resolve())
    if not root.is_dir() or any(root == p or p.is_relative_to(root) for p in private):
        raise ValueError("Choose a project directory, not a home or system directory")
    system = ("/etc", "/private/etc", "/var", "/private/var", "/System", "/Library",
              "/usr", "/bin", "/sbin", "/dev", "/proc", "/sys", "/run")
    if root == Path("/private") or any(root.is_relative_to(Path(p)) for p in system):
        raise ValueError("System directories cannot be used as execution workspaces")
    if any(root.is_relative_to(p) for p in private[1:]):
        raise ValueError("Rune's private data cannot be mounted into the execution environment")
    if any(char in str(root) for char in (",", "\n")):
        raise ValueError("The workspace path is not supported by Docker's mount syntax")
    if not image or image.startswith("-") or any(char.isspace() for char in image):
        raise ValueError("Configure a valid container image")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key.startswith("DOCKER_") for key in env_names):
        raise ValueError("Invalid or reserved container environment variable")
    argv = ["docker", "run", "--rm", "--pull=never", "--name", name, "--init",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", "--memory=2g", "--cpus=2", "--network", "bridge" if network else "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--env", "HOME=/tmp",
            "--mount", f"type=bind,src={root},dst={root}", "--workdir", str(workdir)]
    for key in env_names:
        argv.extend(("--env", key))
    return [*argv, "--entrypoint", "/bin/sh", image, "-c", command]


async def remove_container(name: str, env: dict[str, str]) -> bool:
    try:
        cleanup = await asyncio.create_subprocess_exec(
            "docker", "rm", "--force", name, env=env,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(cleanup.wait(), 5)
        except TimeoutError:
            cleanup.kill()
            await cleanup.wait()
            log.error("container_cleanup_timeout", container=name)
            return False
        return cleanup.returncode == 0
    except OSError as exc:
        log.error("container_cleanup_failed", container=name, error=str(exc))
        return False


async def execute_container(params, config) -> CapabilityResult:
    if params.mode != "oneshot":
        return CapabilityResult(success=False, error="Container execution supports oneshot commands only.",
                                metadata={"action_status": "not_executed"})
    if shutil.which("docker") is None:
        return CapabilityResult(success=False, error="Docker is unavailable; the command was not run.",
                                metadata={"action_status": "not_executed"})
    name = f"rune-command-{uuid4().hex}"
    proc = capture = None
    env = child_environment(params.env)
    # Use the user's Docker context, never a tool-supplied daemon address.
    env.pop("DOCKER_HOST", None)
    env.pop("DOCKER_CONTEXT", None)
    try:
        from rune.safety.execution_environment import execution_workspace
        argv = container_argv(params.command, params.cwd or os.getcwd(), image=config.image,
                              name=name, network=config.allow_network, env_names=tuple(params.env or {}),
                              workspace=execution_workspace(params.cwd))
        proc = await asyncio.create_subprocess_exec(
            *argv, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        capture = asyncio.create_task(capture_output(proc, 100_000))
        stdout, stderr = await asyncio.wait_for(asyncio.shield(capture), params.timeout / 1000)
        return CapabilityResult(
            success=proc.returncode == 0, output=stdout,
            error=stderr if proc.returncode else None,
            metadata={"exit_code": proc.returncode, "execution_environment": "container",
                      "action_status": "completed" if proc.returncode == 0 else "unknown"},
        )
    except TimeoutError:
        return CapabilityResult(success=False, error="Container command timed out; inspect its workspace before retrying.",
                                metadata={"action_status": "unknown", "timeout": True})
    except (OSError, ValueError) as exc:
        return CapabilityResult(success=False, error=str(exc),
                                metadata={"action_status": "unknown" if proc else "not_executed"})
    finally:
        if proc is not None:
            # Killing the Docker client alone would leave the container running.
            try:
                cleanup = await asyncio.create_subprocess_exec(
                    "docker", "rm", "--force", name, env=env,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    await asyncio.wait_for(cleanup.wait(), 5)
                except TimeoutError:
                    cleanup.kill()
                    await cleanup.wait()
                    log.error("container_cleanup_timeout", container=name)
            except OSError as exc:
                log.error("container_cleanup_failed", container=name, error=str(exc))
            finally:
                if capture is not None and (proc.returncode is None or not capture.done()):
                    await stop_capture(proc, capture)
