"""Restore a pending action, then obtain a fresh approval before dispatch."""

from __future__ import annotations

import json
from copy import deepcopy
from uuid import uuid4

from rune.agent.execution_journal import RecoveryBlocked, continuation_action
from rune.safety.approval_request import check_revisions, request_scope


def pending_action(run: dict) -> dict | None:
    for interaction in reversed(run.get("interactions", [])):
        if interaction["kind"] != "approval" or interaction["status"] not in {"interrupted", "suspended"}:
            continue
        request = interaction["request"]
        action = request.get("action")
        if not isinstance(action, dict) or action.get("version") != 1:
            continue
        # Element references and desktop observations expire with their run.
        if action["tool"] == "harness_check" or action["tool"].startswith(("browser_", "desktop_")):
            continue
        return deepcopy(action)
    return None


async def resume_approval(run: dict, journal, approve, emit) -> list[dict]:
    action = pending_action(run)
    if action is None:
        return []
    from rune.capabilities.registry import get_capability_registry
    from rune.safety.approval_context import approval_granted

    tool, params = action["tool"], action["params"]
    check_revisions(action.get("revisions", {}))
    with request_scope(action):
        accepted = await approve(tool, "Review the interrupted action before continuing:\n"
                                 + json.dumps(params, ensure_ascii=False))
    if not accepted:
        raise RecoveryBlocked("The interrupted action was not approved; no action was dispatched.")
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
    # Let the next model step retrieve this receipt without repeating the action.
    journal.previous = [*(journal.previous or []), *records]
    return records
