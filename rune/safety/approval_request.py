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


async def execute_approved_action(action: dict, journal, emit) -> list[dict]:
    """Run the approved action and save its result for continuation."""
    from rune.agent.execution_journal import RecoveryBlocked, continuation_action
    from rune.capabilities.registry import get_capability_registry
    from rune.safety.approval_context import approval_granted

    tool, params = action["tool"], action["params"]
    check_revisions(action.get("revisions", {}))
    call_id = action.get("callId") or f"resumed:{uuid4().hex}"
    await emit("tool_call", {"runId": journal.run_id, "callId": call_id, "toolName": tool, "args": params})
    with continuation_action(call_id), approval_granted(tool, params, revisions=action.get("revisions", {})):
        result = await get_capability_registry().execute(tool, params)
    await emit("tool_result", {"runId": journal.run_id, "callId": call_id, "toolName": tool,
                                "success": result.success, "result": result.output or result.error or "",
                                **({"fileChange": result.metadata["fileChange"]}
                                   if (result.metadata or {}).get("fileChange") else {})})
    if not result.success:
        raise RecoveryBlocked(result.error or "The interrupted action could not be completed.")
    records = journal.store.attempts(journal.run_id)
    journal.previous = [*(journal.previous or []), *records]
    return records
