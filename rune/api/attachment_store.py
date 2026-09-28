"""Conversation-owned uploads, separate from run snapshots and event payloads."""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rune.agent.document_attachments import (
    DOCUMENT_SUFFIXES,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    MAX_TOTAL_BYTES,
)

IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


@dataclass
class Upload:
    info: dict[str, Any]
    content: bytes | None = None


class AttachmentStore:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def prepare(self, session_id: str, attachments: list[dict[str, Any]]) -> list[Upload]:
        if not isinstance(attachments, list) or any(not isinstance(item, dict) for item in attachments):
            raise ValueError("Attachments must be a list of files.")
        if len(attachments) > MAX_ATTACHMENTS:
            raise ValueError("At most 10 files can be attached to a message.")
        uploads: list[Upload] = []
        total = 0
        for item in attachments:
            name = os.path.basename(str(item.get("name") or "attachment"))[:255]
            mime = str(item.get("mimeType") or "")
            ref = item.get("ref")
            if ref:
                if item.get("data"):
                    raise ValueError("Supply an attachment reference or file data, not both.")
                row = self.db.execute(
                    "SELECT ref, name, mime, digest, size FROM web_attachments WHERE session_id = ? AND ref = ?",
                    (session_id, ref),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Reattach {name}: the file is not available in this conversation.")
                if (name, mime) != (row[1], row[2]):
                    raise ValueError("The attachment reference does not match its name or type.")
                upload = Upload(dict(zip(("ref", "name", "mimeType", "digest", "size"), row, strict=True)))
            else:
                data = item.get("data") or ""
                if not isinstance(data, str) or len(data) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
                    raise ValueError(f"{name}: over the 20MB file limit.")
                if mime not in IMAGE_MIMES and Path(name).suffix.lower() not in DOCUMENT_SUFFIXES:
                    raise ValueError(f"{name}: unsupported attachment type.")
                try:
                    content = base64.b64decode(data, validate=True)
                except ValueError as exc:
                    raise ValueError(f"{name}: file data was corrupted in transit.") from exc
                if not content or len(content) > MAX_ATTACHMENT_BYTES:
                    raise ValueError(f"{name}: empty file or over the 20MB file limit.")
                digest = hashlib.sha256(content).hexdigest()
                existing = self.db.execute(
                    "SELECT ref FROM web_attachments WHERE session_id = ? AND digest = ? AND name = ? AND mime = ?",
                    (session_id, digest, name, mime),
                ).fetchone()
                duplicate = next((u for u in uploads if (u.info["digest"], u.info["name"], u.info["mimeType"]) == (digest, name, mime)), None)
                ref = existing[0] if existing else duplicate.info["ref"] if duplicate else uuid4().hex
                upload = Upload({"ref": ref, "name": name, "mimeType": mime, "digest": digest, "size": len(content)},
                                None if existing or duplicate else content)
            total += upload.info["size"]
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Attachments exceed the 40MB message limit.")
            uploads.append(upload)
        return uploads

    def commit(self, session_id: str, uploads: list[Upload]) -> None:
        # The caller commits uploads and the accepted run in one transaction.
        for upload in uploads:
            if upload.content is not None:
                info = upload.info
                self.db.execute("INSERT INTO web_attachments VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (info["ref"], session_id, info["name"], info["mimeType"],
                                 info["digest"], info["size"], upload.content))

    def hydrate(self, session_id: str, attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for item in attachments:
            if not item.get("ref"):
                result.append(item)  # Runs saved before references were introduced.
                continue
            row = self.db.execute("SELECT content, digest FROM web_attachments WHERE session_id = ? AND ref = ?",
                                  (session_id, item["ref"])).fetchone()
            if row is None:
                raise ValueError(f"Reattach {item.get('name', 'file')}: the saved file is unavailable.")
            if hashlib.sha256(row[0]).hexdigest() != row[1]:
                raise ValueError(f"Reattach {item.get('name', 'file')}: the saved file failed its integrity check.")
            result.append({**item, "data": base64.b64encode(row[0]).decode("ascii")})
        return result


def attachment_refs(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item[key] for key in ("name", "mimeType", "ref", "size") if key in item} for item in attachments]
