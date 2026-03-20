"""FileTool - file system operations.

Ported from src/tools/file.ts.  Supports scan, list, read, create,
createDirs, move, copy, and soft-delete (trash).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.tools.base import Tool
from rune.types import Domain, RiskLevel, ToolResult
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Trash settings (mirrors TS TRASH_DIR / TRASH_RETENTION_DAYS)
_TRASH_DIR = Path.home() / ".rune" / "data" / "trash"
_TRASH_RETENTION_DAYS = 7


def cleanup_trash(retention_days: int = _TRASH_RETENTION_DAYS) -> int:
    """Remove trash entries older than *retention_days*. Returns count removed."""
    if not _TRASH_DIR.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for entry in _TRASH_DIR.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("trash_cleanup", removed=removed, retention_days=retention_days)
    return removed

# Deny-patterns (mirrors defaultPolicy.file.denyPatterns)
_DENY_PATTERNS: list[str] = [
    ".env",
    ".ssh",
    ".gnupg",
    ".aws/credentials",
    "id_rsa",
    "id_ed25519",
]

# System paths that must never be touched
_SYSTEM_PATHS: list[str] = [
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/System",
    "/Library",
    "/etc",
    "/var",
]


def _is_system_path(p: str) -> bool:
    resolved = os.path.realpath(p)
    # Direct containment: path is inside a system path
    if any(resolved == sp or resolved.startswith(sp + "/") for sp in _SYSTEM_PATHS):
        return True
    # Reverse containment: path is a PARENT of a system path (e.g. "/")
    if any(sp.startswith(resolved + "/") for sp in _SYSTEM_PATHS):
        return True
    # Root or near-root paths (depth < 3: "/", "/Users", "/tmp", etc.)
    return bool(resolved == "/" or len(Path(resolved).parts) < 3)


def _matches_deny_pattern(p: str, pattern: str) -> bool:
    return pattern in p


class FileTool(Tool):
    """File system operations (scan, list, read, create, move, copy, delete)."""

    @property
    def name(self) -> str:
        return "file"

    @property
    def domain(self) -> Domain:
        return Domain.FILE

    @property
    def description(self) -> str:
        return "File system operations (scan, list, read, create, move, copy, delete)"

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM

    @property
    def actions(self) -> list[str]:
        return ["scan", "list", "read", "create", "create_dirs", "move", "copy", "delete"]

    # -- validate -----------------------------------------------------------

    async def validate(self, params: dict[str, Any]) -> tuple[bool, str]:
        action = params.get("action", "")
        if not action:
            return False, "Missing action parameter"
        if action not in self.actions:
            return False, f"Unknown action: {action}"

        # Check all path parameters
        for key in ("path", "source", "destination", "base_path"):
            value = params.get(key)
            # Reject empty strings - they resolve to CWD and bypass validation
            if key in ("path", "source", "destination"):
                if isinstance(value, str) and not value.strip():
                    return False, f"Empty {key} parameter"
            if value and isinstance(value, str):
                expanded = os.path.expanduser(value)
                if _is_system_path(expanded):
                    return False, f"System path access denied: {value}"
                for pattern in _DENY_PATTERNS:
                    if _matches_deny_pattern(expanded, pattern):
                        return False, f"Sensitive path access denied: {value}"

        return True, ""

    # -- simulate -----------------------------------------------------------

    async def simulate(self, params: dict[str, Any]) -> ToolResult:
        action = params.get("action", "")
        # Read-only actions can run directly
        if action in ("scan", "list", "read"):
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
            if action == "scan":
                return await self._scan(params)
            elif action == "list":
                return await self._list(params)
            elif action == "read":
                return await self._read(params)
            elif action == "create":
                return await self._create(params)
            elif action == "create_dirs":
                return await self._create_dirs(params)
            elif action == "move":
                return await self._move(params)
            elif action == "copy":
                return await self._copy(params)
            elif action == "delete":
                return await self._delete(params)
            else:
                return self.failure(f"Unknown action: {action}")
        except Exception as exc:
            return self.failure(f"File action failed: {exc}")

    # -- rollback -----------------------------------------------------------

    async def rollback(self, rollback_data: dict[str, Any]) -> ToolResult:
        action = rollback_data.get("action", "")
        try:
            if action == "create":
                path = rollback_data.get("path", "")
                if path and os.path.exists(path):
                    os.remove(path)
                    return self.success(data={"rolled_back": "create", "path": path})
            elif action == "move":
                src = rollback_data.get("original_source", "")
                dst = rollback_data.get("original_destination", "")
                if src and dst and os.path.exists(dst):
                    shutil.move(dst, src)
                    return self.success(data={"rolled_back": "move"})
            elif action == "delete":
                trash_path = rollback_data.get("trash_path", "")
                original_path = rollback_data.get("original_path", "")
                if trash_path and original_path and os.path.exists(trash_path):
                    shutil.move(trash_path, original_path)
                    return self.success(data={"rolled_back": "delete"})
            return self.failure(f"Cannot rollback action: {action}")
        except Exception as exc:
            return self.failure(f"Rollback failed: {exc}")

    # -- action implementations ---------------------------------------------

    async def _scan(self, params: dict[str, Any]) -> ToolResult:
        """Glob-based file scan."""
        import glob as glob_mod

        base = os.path.expanduser(params.get("path", "."))
        pattern = params.get("pattern", "**/*")
        full_pattern = os.path.join(base, pattern)
        matches = glob_mod.glob(full_pattern, recursive=True)
        return self.success(data={"files": matches[:500], "total": len(matches)})

    async def _list(self, params: dict[str, Any]) -> ToolResult:
        """List directory contents."""
        path = Path(os.path.expanduser(params.get("path", ".")))
        if not path.is_dir():
            return self.failure(f"Not a directory: {path}")
        entries = []
        for entry in sorted(path.iterdir()):
            stat = entry.stat()
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "is_directory": entry.is_dir(),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return self.success(data={"entries": entries, "count": len(entries)})

    async def _read(self, params: dict[str, Any]) -> ToolResult:
        """Read file contents."""
        path = Path(os.path.expanduser(params.get("path", "")))
        if not path.is_file():
            return self.failure(f"File not found: {path}")

        max_size = params.get("max_size", 1_048_576)  # 1 MB default
        size = path.stat().st_size
        if size > max_size:
            return self.failure(
                f"File too large ({size} bytes > {max_size} byte limit)"
            )

        content = path.read_text(errors="replace")
        return self.success(data={"content": content, "path": str(path), "size": size})

    async def _create(self, params: dict[str, Any]) -> ToolResult:
        """Create or overwrite a file."""
        path = Path(os.path.expanduser(params.get("path", "")))
        content = params.get("content", "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return self.success(
            data={"path": str(path), "size": len(content)},
            rollback_data={"action": "create", "path": str(path)},
        )

    async def _create_dirs(self, params: dict[str, Any]) -> ToolResult:
        """Create directory tree."""
        path = Path(os.path.expanduser(params.get("path", "")))
        path.mkdir(parents=True, exist_ok=True)
        return self.success(data={"path": str(path)})

    async def _move(self, params: dict[str, Any]) -> ToolResult:
        """Move / rename a file or directory."""
        src = os.path.expanduser(params.get("source", ""))
        dst = os.path.expanduser(params.get("destination", ""))
        if not os.path.exists(src):
            return self.failure(f"Source not found: {src}")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        return self.success(
            data={"source": src, "destination": dst},
            rollback_data={"action": "move", "original_source": src, "original_destination": dst},
        )

    async def _copy(self, params: dict[str, Any]) -> ToolResult:
        """Copy a file or directory."""
        src = os.path.expanduser(params.get("source", ""))
        dst = os.path.expanduser(params.get("destination", ""))
        if not os.path.exists(src):
            return self.failure(f"Source not found: {src}")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return self.success(data={"source": src, "destination": dst})

    async def _delete(self, params: dict[str, Any]) -> ToolResult:
        """Soft-delete: move to trash with metadata."""
        path_str = os.path.expanduser(params.get("path", ""))
        if not os.path.exists(path_str):
            return self.failure(f"Path not found: {path_str}")

        # Prepare trash destination
        _TRASH_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        trash_name = f"{ts}_{os.path.basename(path_str)}"
        trash_path = _TRASH_DIR / trash_name

        shutil.move(path_str, str(trash_path))
        log.info("file_trashed", original=path_str, trash=str(trash_path))

        return self.success(
            data={"path": path_str, "trash_path": str(trash_path)},
            rollback_data={
                "action": "delete",
                "original_path": path_str,
                "trash_path": str(trash_path),
            },
        )
