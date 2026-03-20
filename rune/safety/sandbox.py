"""OS-level command sandbox for RUNE.

Ported 1:1 from src/safety/sandbox.ts - macOS Seatbelt + Linux bubblewrap.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Types

@dataclass(slots=True)
class SandboxConfig:
    enabled: bool = True
    allow_network: bool = False
    writable_paths: list[str] = field(default_factory=list)
    readable_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=list)
    timeout: int = 30_000  # ms


@dataclass(slots=True)
class SandboxResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    killed: bool = False
    error: str | None = None


# Default configuration

def get_default_config() -> SandboxConfig:
    home = str(Path.home())
    workspace = os.getcwd()
    tmp = tempfile.gettempdir()

    return SandboxConfig(
        enabled=True,
        allow_network=False,
        writable_paths=[workspace, os.path.join(home, ".rune"), tmp],
        readable_paths=[
            "/usr", "/bin", "/lib", "/lib64",
            "/System/Library", "/opt/homebrew",
            home,
        ],
        blocked_paths=[
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".env"),
            "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        ],
        timeout=30_000,
    )


def _expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


# macOS Seatbelt profile

def _create_seatbelt_profile(config: SandboxConfig) -> str:
    tmp = tempfile.gettempdir()

    write_rules = "\n    ".join(
        f'(allow file-read* file-write* (subpath "{_expand_path(p)}"))'
        for p in config.writable_paths
    )
    read_rules = "\n    ".join(
        f'(allow file-read* (subpath "{_expand_path(p)}"))'
        for p in config.readable_paths
    )
    block_rules = "\n    ".join(
        f'(deny file-read* file-write* (subpath "{_expand_path(p)}"))'
        for p in config.blocked_paths
    )
    network_rule = "(allow network*)" if config.allow_network else "(deny network*)"

    return f"""(version 1)
(deny default)

;; Import base system profile
(import "/System/Library/Sandbox/Profiles/bsd.sb")

;; Basic process operations
(allow process-fork)
(allow process-exec)
(allow signal (target self))

;; Block sensitive paths FIRST (higher priority)
{block_rules}

;; Allow read-write to workspace and temp
{write_rules}

;; Allow read-only to system paths
{read_rules}

;; Device access
(allow file-read* (subpath "/dev"))
(allow file-write* (literal "/dev/null"))
(allow file-write* (literal "/dev/tty"))

;; Temp directory
(allow file-read* file-write* (subpath "{tmp}"))

;; Network access
{network_rule}

;; Allow mach services needed for basic operation
(allow mach-lookup
    (global-name "com.apple.system.logger")
    (global-name "com.apple.system.notification_center"))
"""


# Sandbox Execution

async def _execute_macos_sandbox(
    command: str,
    args: list[str],
    config: SandboxConfig,
) -> SandboxResult:
    profile = _create_seatbelt_profile(config)
    profile_path = os.path.join(tempfile.gettempdir(), f"rune-sandbox-{int(time.time() * 1000)}.sb")

    try:
        Path(profile_path).write_text(profile, encoding="utf-8")

        timeout_sec = config.timeout / 1000.0
        env = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"}

        try:
            proc = await asyncio.create_subprocess_exec(
                "sandbox-exec", "-f", profile_path, command, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    success=False, exit_code=None, killed=True,
                    error="Sandbox timeout",
                )

            return SandboxResult(
                success=proc.returncode == 0,
                stdout=stdout_bytes.decode(errors="replace"),
                stderr=stderr_bytes.decode(errors="replace"),
                exit_code=proc.returncode,
            )

        except FileNotFoundError:
            return SandboxResult(
                success=False, error="sandbox-exec not found",
            )

    finally:
        with contextlib.suppress(OSError):
            os.unlink(profile_path)


async def _execute_linux_sandbox(
    command: str,
    args: list[str],
    config: SandboxConfig,
) -> SandboxResult:
    if not shutil.which("bwrap"):
        return SandboxResult(success=False, error="bubblewrap is not available")

    bwrap_args: list[str] = [
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
    ]

    if config.allow_network:
        bwrap_args.append("--share-net")

    # Read-only binds
    for p in config.readable_paths:
        expanded = _expand_path(p)
        if os.path.exists(expanded):
            bwrap_args.extend(["--ro-bind", expanded, expanded])

    # Read-write binds
    for p in config.writable_paths:
        expanded = _expand_path(p)
        if os.path.exists(expanded):
            bwrap_args.extend(["--bind", expanded, expanded])

    # System paths
    bwrap_args.extend([
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--symlink", "usr/lib", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ])

    bwrap_args.extend([command, *args])

    timeout_sec = config.timeout / 1000.0
    env = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}

    try:
        proc = await asyncio.create_subprocess_exec(
            "bwrap", *bwrap_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                success=False, exit_code=None, killed=True,
                error="Sandbox timeout",
            )

        return SandboxResult(
            success=proc.returncode == 0,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            exit_code=proc.returncode,
        )

    except FileNotFoundError:
        return SandboxResult(success=False, error="bwrap not found")


# Public API

_sandbox_support_cache: dict[str, tuple[bool, float]] = {}
_CACHE_TTL = 30.0


async def has_sandbox_support() -> bool:
    """Check if sandbox is available on this platform (30s cache)."""
    now = time.monotonic()
    cached = _sandbox_support_cache.get("support")
    if cached and (now - cached[1]) < _CACHE_TTL:
        return cached[0]

    system = platform.system()
    if system == "Darwin":
        supported = shutil.which("sandbox-exec") is not None
    elif system == "Linux":
        supported = shutil.which("bwrap") is not None
    else:
        supported = False

    _sandbox_support_cache["support"] = (supported, now)
    return supported


async def execute_sandboxed(
    command: str,
    args: list[str] | None = None,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute a command inside an OS-level sandbox."""
    args = args or []
    config = config or get_default_config()

    system = platform.system()
    if system == "Darwin":
        return await _execute_macos_sandbox(command, args, config)
    elif system == "Linux":
        return await _execute_linux_sandbox(command, args, config)
    else:
        return SandboxResult(
            success=False,
            error=f"Sandbox not supported on {system}",
        )


async def execute_shell_sandboxed(
    shell_command: str,
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute a shell command string inside a sandbox."""
    return await execute_sandboxed("/bin/sh", ["-c", shell_command], config)
