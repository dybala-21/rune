"""Render a document bundle inside its staging directory."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.process_worker import receive_message, send_message, watch_parent

log = get_logger(__name__)


def render_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    from rune.capabilities.bundle_data import calculate, read_table
    from rune.capabilities.bundle_revision import digest
    from rune.capabilities.document import _RENDERERS
    from rune.capabilities.document_bundle import (
        DocumentBundleParams,
        _prepare_documents,
        _verify_file,
    )

    params = DocumentBundleParams.model_validate(payload["spec"])
    stage = Path(payload["stage"])
    source = stage / "source.snapshot"
    rows = read_table(source.read_bytes(), Path(params.source_path).suffix.lower(), params.sheet)
    values, display, selected = calculate(rows, params.filters, params.metrics)
    documents = _prepare_documents(params, values, display)
    artifacts = []
    for doc in documents:
        path = stage / "version" / doc.path
        _RENDERERS[doc.format][0](path, doc)
        _verify_file(path, doc)
        artifacts.append({"filename": doc.path, "format": doc.format, "sha256": digest(path),
                          "bytes": path.stat().st_size})
    return {"rows_total": len(rows), "rows_selected": selected, "metrics": values,
            "display_values": display, "artifacts": artifacts}


def serve(sock: socket.socket) -> None:
    watch_parent()
    try:
        request = receive_message(sock, None)
        try:
            result = {"success": True, "data": render_snapshot(request["payload"])}
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        send_message(sock, {"id": request["id"], "result": result}, None)
    except (EOFError, OSError):
        log.debug("document_worker_connection_closed")
        return
    finally:
        sock.close()
