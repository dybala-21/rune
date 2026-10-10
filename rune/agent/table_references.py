"""Snapshot request-owned reference documents before fixing a table contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rune.agent.provenance import ArtifactLedger
from rune.capabilities.document_inspection import inspect_bytes, read_snapshot

_DOCUMENTS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".html", ".docx", ".pdf", ".pptx"}


def read_references(request: str, prior: list[str], workspace: str, roles: dict[str, str]) -> list[dict]:
    ledger = ArtifactLedger.for_request("\n".join([*prior, request]), root=workspace)
    documents = []
    for name, paths in sorted(ledger.requested_paths.items()):
        if roles.get(name) != "input" or Path(name).suffix.lower() not in _DOCUMENTS:
            continue
        for locator in sorted(paths):
            source = Path(locator).expanduser()
            if not source.is_absolute():
                source = Path(workspace) / source
            path, data = read_snapshot(source)
            if any(document["path"] == str(path) for document in documents):
                continue
            result = inspect_bytes(data, path.suffix.lower().lstrip("."), 16_000)
            if result["truncated"] or sum(len(d["text"]) for d in documents) + len(result["text"]) > 16_000:
                raise ValueError("Reference documents exceed the table requirement context limit; use a smaller policy document")
            if not result["text"].strip() or result["facts"].get("pages_without_text"):
                raise ValueError(f"Reference document could not be read completely: {path.name}")
            documents.append({"path": str(path), "locator": str(source.absolute()),
                              "sha256": result["facts"]["sha256"], "text": result["text"],
                              "extraction_warnings": result["warnings"]})
            if len(documents) > 8:
                raise ValueError("Table requirements support at most eight reference documents")
    return documents


def reference_revisions(documents: list[dict]) -> list[dict]:
    return [{key: document[key] for key in ("path", "locator", "sha256")} for document in documents]


def check_references(revisions: list[dict]) -> None:
    for document in revisions:
        path, data = read_snapshot(document["locator"])
        if str(path) != document["path"] or hashlib.sha256(data).hexdigest() != document["sha256"]:
            raise ValueError("Reference document changed after requirements were fixed; start a new request")
