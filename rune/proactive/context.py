"""Proactive context gathering for RUNE.

Collects workspace, git, process, and temporal context to inform
proactive suggestion generation.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.proactive.activity_mode import ActivityMode
from rune.utils.logger import get_logger

log = get_logger(__name__)

_gatherer: ContextGatherer | None = None


@dataclass(slots=True)
class AwarenessContext:
    """Snapshot of the current environment and user state."""

    workspace_root: str = ""
    recent_files: list[str] = field(default_factory=list)
    git_status: str = ""
    running_processes: list[str] = field(default_factory=list)
    time_context: dict[str, Any] = field(default_factory=dict)
    user_activity_mode: ActivityMode = "exploration"


class ContextGatherer:
    """Gathers environmental context for proactive awareness."""

    __slots__ = ("_workspace_root",)

    def __init__(self, workspace_root: str | None = None) -> None:
        self._workspace_root = workspace_root or os.getcwd()

    async def gather(self) -> AwarenessContext:
        """Collect all context signals asynchronously.

        Returns:
            A populated AwarenessContext snapshot.
        """
        git_status, recent_files, running_processes = await asyncio.gather(
            self._get_git_status(),
            asyncio.to_thread(self._get_recent_files),
            asyncio.to_thread(self._get_running_processes),
        )

        now = datetime.now()
        time_context: dict[str, Any] = {
            "hour": now.hour,
            "weekday": now.strftime("%A"),
            "is_weekend": now.weekday() >= 5,
            "iso": now.isoformat(),
        }

        ctx = AwarenessContext(
            workspace_root=self._workspace_root,
            recent_files=recent_files,
            git_status=git_status,
            running_processes=running_processes,
            time_context=time_context,
        )

        log.debug(
            "context_gathered",
            files=len(recent_files),
            has_git=bool(git_status),
            processes=len(running_processes),
        )
        return ctx

    async def _get_git_status(self) -> str:
        """Run ``git status --short`` in the workspace root."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--short",
                cwd=self._workspace_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="replace").strip()
        except (TimeoutError, OSError) as exc:
            log.debug("git_status_failed", error=str(exc))
            return ""

    def _get_recent_files(self) -> list[str]:
        """Return recently modified files in the workspace root (up to 20)."""
        root = Path(self._workspace_root)
        if not root.is_dir():
            return []

        files: list[tuple[float, str]] = []
        try:
            for entry in root.rglob("*"):
                if entry.is_file() and not any(
                    part.startswith(".") or part == "node_modules" or part == "__pycache__"
                    for part in entry.parts
                ):
                    try:
                        mtime = entry.stat().st_mtime
                        files.append((mtime, str(entry.relative_to(root))))
                    except OSError:
                        continue
                if len(files) > 500:
                    break
        except OSError:
            pass

        files.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in files[:20]]

    def _get_running_processes(self) -> list[str]:
        """Return a list of notable running process names."""
        # Lightweight heuristic: check common dev servers
        notable = []
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-eo", "comm"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                processes = set(result.stdout.strip().splitlines())
                dev_keywords = {"node", "python", "cargo", "go", "java", "gradle", "webpack"}
                for proc in processes:
                    name = proc.strip().split("/")[-1].lower()
                    if any(kw in name for kw in dev_keywords):
                        notable.append(proc.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass

        return notable[:10]


def get_context_gatherer(workspace_root: str | None = None) -> ContextGatherer:
    """Get or create the singleton ContextGatherer."""
    global _gatherer
    if _gatherer is None:
        _gatherer = ContextGatherer(workspace_root)
    return _gatherer
