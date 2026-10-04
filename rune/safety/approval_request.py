"""Typed approval requests, including the file revision being approved."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from uuid import uuid4

_request: ContextVar[dict | None] = ContextVar("approval_request", default=None)


def action_request(tool: str, params: dict) -> dict:
    from rune.agent.execution_journal import fingerprint
    from rune.agent.loop import current_tool_call_id

    revisions = {}
    if tool in {"file_write", "file_edit", "file_delete", "document_create"} and params.get("path"):
        revisions[params["path"]] = fingerprint(params["path"])
    return {"version": 1, "tool": tool, "params": deepcopy(params), "revisions": revisions,
            "callId": current_tool_call_id() or uuid4().hex}


def check_revisions(revisions: dict) -> None:
    from rune.agent.execution_journal import RecoveryBlocked, fingerprint

    for path, expected in revisions.items():
        if fingerprint(path) != expected:
            raise RecoveryBlocked(f"The file changed while approval was pending: {path}. Read it again before requesting approval.")


def current_request() -> dict | None:
    return deepcopy(_request.get())


@contextmanager
def request_scope(action: dict):
    token = _request.set(deepcopy(action))
    try:
        yield
    finally:
        _request.reset(token)
