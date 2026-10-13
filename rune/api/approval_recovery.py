"""Restore a pending action, then obtain a fresh approval before dispatch."""

from __future__ import annotations

import json
from copy import deepcopy

from rune.agent.execution_journal import RecoveryBlocked
from rune.safety.approval_request import check_revisions, execute_approved_action, request_scope


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
    tool, params = action["tool"], action["params"]
    check_revisions(action.get("revisions", {}))
    with request_scope(action):
        accepted = await approve(tool, "Review the interrupted action before continuing:\n"
                                 + json.dumps(params, ensure_ascii=False))
    if not accepted:
        raise RecoveryBlocked("The interrupted action was not approved; no action was dispatched.")
    return await execute_approved_action(action, journal, emit)
