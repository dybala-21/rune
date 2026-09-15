"""Immutable downloads published by the native host for one conversation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path
from uuid import uuid4

from rune.computer.protocol import DesktopError

MAX_ARTIFACT = 16 * 1024 * 1024


class Artifacts:
    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from rune.utils.paths import rune_home
            root = rune_home() / "artifacts"
        self.root = root

    def directory(self, session_id: str) -> Path:
        return self.root / hashlib.sha256(session_id.encode()).hexdigest()

    def publish(self, session_id: str, run_id: str, path: str, content: bytes, digest: str) -> dict:
        if not session_id or not run_id or not Path(path).is_absolute():
            raise DesktopError("The document is missing its conversation or saved path.")
        if len(content) > MAX_ARTIFACT or hashlib.sha256(content).hexdigest() != digest:
            raise DesktopError("The published document failed its content check.")
        directory = self.directory(session_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        artifact_id = uuid4().hex
        receipt = {"id": artifact_id, "sessionId": session_id, "runId": run_id,
                   "path": unicodedata.normalize("NFC", path), "name": Path(path).name,
                   "sha256": digest, "size": len(content), "scope": "download", "verified": True}
        for suffix, data in (("blob", content), ("json", json.dumps(receipt, ensure_ascii=False).encode())):
            fd = os.open(directory / f"{artifact_id}.{suffix}", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
        return receipt

    def read(self, session_id: str, artifact_id: str) -> tuple[dict, bytes]:
        if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            raise FileNotFoundError("Unknown artifact")
        directory = self.directory(session_id)
        receipt = json.loads(self._read_regular(directory / f"{artifact_id}.json", 16 * 1024))
        if receipt.get("sessionId") != session_id or receipt.get("id") != artifact_id:
            raise FileNotFoundError("Unknown artifact")
        content = self._read_regular(directory / f"{artifact_id}.blob", MAX_ARTIFACT)
        if len(content) != receipt["size"] or hashlib.sha256(content).hexdigest() != receipt["sha256"]:
            raise DesktopError("The download changed after publication. Publish the document again.")
        return receipt, content

    def for_path(self, session_id: str, path: str) -> tuple[dict, bytes] | None:
        expected = unicodedata.normalize("NFC", path)
        matches = []
        for manifest in self.directory(session_id).glob("*.json"):
            receipt = json.loads(self._read_regular(manifest, 16 * 1024))
            if receipt.get("path") == expected:
                matches.append((manifest.stat().st_mtime_ns, manifest.stem))
        if not matches:
            return None
        return self.read(session_id, max(matches)[1])

    @staticmethod
    def _read_regular(path: Path, limit: int) -> bytes:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise DesktopError("The published artifact is not a bounded regular file.")
            content = file.read(limit + 1)
            if len(content) > limit:
                raise DesktopError("The published artifact exceeds the download limit.")
            return content
