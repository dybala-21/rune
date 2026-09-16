"""Keep unsuccessful file mutations out of completion evidence."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)
_MAX_REVISION_BYTES = 16 * 1024 * 1024

FILE_MUTATIONS = frozenset({
    "file_write", "file_edit", "file_delete", "document_create",
    "document_bundle", "document_bundle_update",
})


def may_have_changed(result: CapabilityResult) -> bool:
    metadata = result.metadata or {}
    return (metadata.get("action_status") != "not_executed"
            and not metadata.get("requires_approval")
            and metadata.get("changed") is not False)


@dataclass(frozen=True)
class FailedMutation:
    tool: str
    path: str
    status: str
    expected_hash: str | None = None


class FileOutcomes:
    def __init__(self, workspace: str = "") -> None:
        self.workspace = Path(workspace or Path.cwd())
        self.pending: dict[tuple[str, str], FailedMutation] = {}

    def _path(self, params: dict[str, Any]) -> Path | None:
        raw = params.get("file_path") or params.get("path") or params.get("directory")
        return self.workspace / Path(raw).expanduser() if isinstance(raw, str) and raw else None

    @staticmethod
    def _key(path: Path) -> str:
        try:
            return str(path.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            log.debug("file_outcome_path_unresolved", path=str(path), error=str(exc))
            return os.path.abspath(path)

    @staticmethod
    def _read(path: Path) -> bytes | None:
        from rune.safety.guardian import get_guardian

        try:
            if not get_guardian().validate_file_read_path(str(path)).allowed:
                return None
            if not path.is_file() or path.stat().st_size > _MAX_REVISION_BYTES:
                return None
            with path.open("rb") as stream:
                data = stream.read(_MAX_REVISION_BYTES + 1)
            return data if len(data) <= _MAX_REVISION_BYTES else None
        except (OSError, RuntimeError, ValueError) as exc:
            log.debug("file_outcome_unreadable", path=str(path), error=str(exc))
            return None

    def expected_edit(self, tool: str, params: dict[str, Any]) -> str | None:
        if tool != "file_edit" or (path := self._path(params)) is None:
            return None
        data = self._read(path)
        if data is None:
            return None
        try:
            before = data.decode("utf-8")
            search, replacement = params.get("search"), params.get("replace")
            if not isinstance(search, str) or not isinstance(replacement, str) or not search or search not in before:
                return None
            after = before.replace(search, replacement, -1 if params.get("all") else 1)
            return hashlib.sha256(after.encode("utf-8")).hexdigest()
        except UnicodeError as exc:
            log.debug("file_outcome_not_text", path=str(path), error=str(exc))
            return None

    def observe(self, tool: str, params: dict[str, Any], result: CapabilityResult,
                expected_hash: str | None = None) -> None:
        if tool not in FILE_MUTATIONS:
            return
        path = self._path(params)
        if path is None:
            return
        kind = "delete" if tool == "file_delete" else "write"
        key = self._key(path), kind
        if result.success:
            self.pending.pop(key, None)
        else:
            status = "outcome unknown" if may_have_changed(result) else "not executed"
            if tool == "file_write" and isinstance(params.get("content"), str):
                try:
                    expected_hash = hashlib.sha256(params["content"].encode(params.get("encoding", "utf-8"))).hexdigest()
                except (UnicodeError, LookupError) as exc:
                    log.debug("file_outcome_encoding_failed", error=str(exc))
            self.pending[key] = FailedMutation(tool, str(path), status, expected_hash)

    def reconcile(self) -> list[str]:
        """Confirm a repair by matching file contents or confirming deletion."""
        repaired = []
        for key, failure in list(self.pending.items()):
            path = Path(failure.path)
            if self._key(path) != key[0]:
                continue
            if failure.tool == "file_delete":
                try:
                    path.lstat()
                except FileNotFoundError:
                    repaired.append(failure.path)
                    del self.pending[key]
                except (OSError, ValueError) as exc:
                    log.debug("file_outcome_absence_unconfirmed", path=str(path), error=str(exc))
                continue
            if failure.expected_hash is None:
                continue
            data = self._read(path)
            if data is not None and hashlib.sha256(data).hexdigest() == failure.expected_hash:
                repaired.append(failure.path)
                del self.pending[key]
        return repaired

    def blocker(self) -> str:
        if not self.pending:
            return ""
        details = "; ".join(
            f"{failure.tool} {failure.path}: {failure.status}"
            for failure in list(self.pending.values())[:5]
        )
        return (
            "File operations remain unconfirmed: " + details + ". "
            "A successful operation on another file does not confirm these results. "
            "Resolve the affected file operations before claiming completion. "
            "Do not retry a denied action without authorization; report the blocker "
            "with task_blocked if it prevents completion."
        )
