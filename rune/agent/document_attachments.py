"""Store uploaded documents for the existing file and document tools."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path

from rune.agent.isolation import enforce
from rune.safety.guardian import get_guardian

DOCUMENT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".pdf", ".docx", ".xlsx", ".pptx"}
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_ATTACHMENTS = 10
MAX_TOTAL_BYTES = 40 * 1024 * 1024


def save_document(name: str, data: str, workspace: str, reference: str = "") -> str:
    """Create a private copy without trusting the uploaded filename as a path."""
    suffix = Path(name).suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES:
        raise ValueError("unsupported document type")
    if len(data) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
        raise ValueError("over the 20MB file limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError as error:
        raise ValueError("file data was corrupted in transit") from error
    if not raw or len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError("empty file or over the 20MB file limit")
    root = Path(workspace).expanduser().resolve(strict=True)
    candidate = str(root / f"rune-attachment{suffix}")
    isolation_error = enforce(candidate)
    validation = get_guardian().validate_file_path(candidate)
    if isolation_error or not validation.allowed:
        raise ValueError(isolation_error or validation.reason)
    if re.fullmatch(r"[a-f0-9]{32}", reference):
        path = root / f"rune-attachment-{reference}{suffix}"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Never replace a copy the user has edited or redirected.
            if not path.is_symlink() and path.is_file() and path.stat().st_size == len(raw):
                if hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(raw).digest():
                    return str(path)
        else:
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(raw)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            return str(path)
    fd, path = tempfile.mkstemp(prefix="rune-attachment-", suffix=suffix, dir=root)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return path
