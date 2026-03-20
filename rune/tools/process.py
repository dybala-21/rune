"""ProcessTool - subprocess execution with safety constraints.

Ported from src/tools/process.ts.  Supports list, run, kill, find,
and monitor actions.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from typing import Any

from rune.tools.base import Tool
from rune.types import Domain, RiskLevel, ToolResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Regex-based deny patterns (replaces simple substring matching).
# Each pattern is designed to catch common evasion techniques like
# extra whitespace, split flags, and long-form options.
_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r?|--recursive)\s+/"),
    re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r?|--recursive)\s+/\*"),
    re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|-[a-zA-Z]*f[a-zA-Z]*r?|--recursive)\s+~"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*\bif="),
    re.compile(r":\(\)\s*\{"),  # fork bomb
    re.compile(r"\bchmod\s+(-R\s+)?777\s+/"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bhalt\b"),
    re.compile(r"\binit\s+[06]\b"),
]

# Mirrors defaultPolicy.process.protectedProcesses
_PROTECTED_PROCESSES: list[str] = [
    "systemd",
    "init",
    "launchd",
    "kernel_task",
    "WindowServer",
    "loginwindow",
]

_DEFAULT_TIMEOUT_S = 30


class ProcessTool(Tool):
    """System process management (list, run, kill, find, monitor)."""

    @property
    def name(self) -> str:
        return "process"

    @property
    def domain(self) -> Domain:
        return Domain.PROCESS

    @property
    def description(self) -> str:
        return "System process management (list, run, kill, find, monitor)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    @property
    def actions(self) -> list[str]:
        return ["list", "run", "kill", "find", "monitor"]

    def __init__(self) -> None:
        self._running: dict[str, asyncio.subprocess.Process] = {}

    # -- validate -----------------------------------------------------------

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        action = params.get("action", "")
        if not action:
            return False, "Missing action parameter"
        if action not in self.actions:
            return False, f"Unknown action: {action}"

        if action == "run":
            command = params.get("command", "")
            if not command:
                return False, "Missing command parameter"

            # 1. Regex-based deny patterns
            for pattern in _DENY_PATTERNS:
                if pattern.search(command):
                    return False, f"Command denied by policy: {command}"

            # 2. Route through Guardian for full safety analysis
            try:
                from rune.safety.guardian import get_guardian
                guardian = get_guardian()
                result = guardian.validate(command)
                if not result.allowed:
                    return False, f"Guardian blocked: {result.reason}"
                if result.requires_approval:
                    return False, f"Command requires approval: {result.reason}"
            except Exception as exc:
                # Fail closed - if Guardian errors, deny the command
                log.error("process_tool_guardian_error", error=str(exc))
                return False, f"Safety validation error (fail-closed): {exc}"

        if action == "kill":
            proc_name = params.get("name", "")
            if proc_name and proc_name in _PROTECTED_PROCESSES:
                return False, f"Cannot kill protected process: {proc_name}"

        return True, ""

    # -- simulate -----------------------------------------------------------

    async def simulate(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        if action in ("list", "find", "monitor"):
            return await self.execute(params)
        return self.success(data={
            "simulation": True,
            "action": action,
            "params": params,
            "message": "This action would be executed",
        })

    # -- execute ------------------------------------------------------------

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")

        valid, err = await self.validate(params)
        if not valid:
            return self.failure(err)

        try:
            if action == "list":
                return await self._list_processes()
            elif action == "run":
                return await self._run(params)
            elif action == "kill":
                return await self._kill(params)
            elif action == "find":
                return await self._find(params)
            elif action == "monitor":
                return await self._monitor(params)
            else:
                return self.failure(f"Unknown action: {action}")
        except Exception as exc:
            return self.failure(f"Process action failed: {exc}")

    # -- rollback -----------------------------------------------------------

    async def rollback(self, rollback_data: dict[str, Any]) -> ToolResult:
        pid = rollback_data.get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
                return self.success(data={"killed_pid": pid})
            except (ProcessLookupError, PermissionError) as exc:
                return self.failure(f"Rollback (kill pid {pid}) failed: {exc}")
        return self.failure("No rollback data available")

    # -- action implementations ---------------------------------------------

    async def _list_processes(self) -> ToolResult:
        """List running processes via ``ps``."""
        proc = await asyncio.create_subprocess_exec(
            "ps", "aux",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode(errors="replace").strip().split("\n")
        processes: list[dict[str, Any]] = []
        for line in lines[1:50]:  # limit output
            parts = line.split(None, 10)
            if len(parts) >= 11:
                processes.append({
                    "user": parts[0],
                    "pid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "memory": float(parts[3]),
                    "command": parts[10],
                })
        return self.success(data={"processes": processes, "total": len(lines) - 1})

    async def _run(self, params: dict[str, Any]) -> ToolResult:
        """Run a command in a subprocess."""
        command: str = params["command"]
        cwd: str | None = params.get("cwd")
        timeout: int = params.get("timeout", _DEFAULT_TIMEOUT_S)
        env = {**os.environ, **(params.get("env") or {})}

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return self.failure(f"Command timed out after {timeout}s: {command}")

        exit_code = proc.returncode or 0
        stdout_str = stdout.decode(errors="replace")[:50_000]
        stderr_str = stderr.decode(errors="replace")[:10_000]

        result_data = {
            "command": command,
            "exit_code": exit_code,
            "stdout": stdout_str,
            "stderr": stderr_str,
        }

        if exit_code != 0:
            return ToolResult(
                success=False,
                data=result_data,
                error=f"Command exited with code {exit_code}",
            )

        return self.success(
            data=result_data,
            rollback_data={"pid": proc.pid} if proc.pid else None,
        )

    async def _kill(self, params: dict[str, Any]) -> ToolResult:
        """Kill a process by PID or name."""
        pid = params.get("pid")
        name = params.get("name", "")
        sig = params.get("signal", "SIGTERM")

        sig_num = getattr(signal, sig, signal.SIGTERM)

        if pid:
            os.kill(int(pid), sig_num)
            return self.success(data={"killed_pid": pid, "signal": sig})

        if name:
            # Find PIDs by name via pgrep
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", name,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            pids = [int(p) for p in stdout.decode().strip().split("\n") if p.strip()]
            killed: list[int] = []
            for p in pids:
                try:
                    os.kill(p, sig_num)
                    killed.append(p)
                except (ProcessLookupError, PermissionError):
                    pass
            return self.success(data={"killed_pids": killed, "name": name, "signal": sig})

        return self.failure("Either pid or name is required")

    async def _find(self, params: dict[str, Any]) -> ToolResult:
        """Find processes matching a pattern."""
        pattern = params.get("pattern", "")
        if not pattern:
            return self.failure("Missing pattern parameter")

        proc = await asyncio.create_subprocess_exec(
            "pgrep", "-fl", pattern,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode(errors="replace").strip().split("\n")
        matches: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                matches.append({"pid": int(parts[0]), "command": parts[1]})
        return self.success(data={"matches": matches, "count": len(matches)})

    async def _monitor(self, params: dict[str, Any]) -> ToolResult:
        """Quick snapshot of system resource usage."""
        import platform

        proc = await asyncio.create_subprocess_exec(
            "uptime",
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        uptime_str = stdout.decode(errors="replace").strip()

        return self.success(data={
            "platform": platform.system(),
            "cpu_count": os.cpu_count(),
            "uptime": uptime_str,
        })
