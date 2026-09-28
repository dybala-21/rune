"""Reserve a conversation before publishing a new execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from rune.api.attachment_store import AttachmentStore
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore


class RunBusy(ValueError):
    pass


class RunAdmission:
    def __init__(self, runs: RunSnapshots, store: RunStore, computers) -> None:
        self.runs, self.store, self.computers = runs, store, computers

    async def accept(self, goal: str, session_id: str | None, *, sticky: bool = False,
                     request_id: str = "", attachments: list[dict[str, Any]] | None = None) -> tuple[str, str, bool]:
        from rune.api import conversation_wiring

        manager = conversation_wiring.get_conv_manager()
        if manager is not None:
            session_id = await conversation_wiring.resolve_conversation(manager, session_id, sticky=sticky)
        session_id = session_id or ""
        self.runs.open()
        uploads = AttachmentStore(self.store.db).prepare(session_id, attachments or [])
        payload_hash = ""
        if request_id:
            content = [goal, [{k: u.info[k] for k in ("name", "mimeType", "digest")} for u in uploads]]
            payload_hash = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
            previous = self.store.requested_run(session_id, request_id, payload_hash)
            if previous:
                return previous, session_id, True
        # No await between checking the reservation and creating it.
        if session_id:
            entry = self.computers.entries.get(session_id)
            if self.store.active_for_session(session_id) or (entry is not None and entry.control is not None):
                raise RunBusy("This chat is still working or stopping. Wait for it to finish.")
        run_id = uuid4().hex[:16]
        self.runs.start(run_id, session_id, goal, request=(request_id, payload_hash) if request_id else None, uploads=uploads)
        return run_id, session_id, False
