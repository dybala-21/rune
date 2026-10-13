"""Validate a saved action before continuing a scheduled task."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from rune.agent.execution_journal import RecoveryBlocked, changed_read_paths, reconcile
from rune.proactive.routine import task_fingerprint
from rune.proactive.routine_budget import budget_owner
from rune.proactive.routine_observation import changes, observe
from rune.safety.approval_request import check_revisions
from rune.safety.execution_environment import execution_config


def pending_run(job, store, operation_id: str) -> dict:
    prefix = f"cron:{job.id}:"
    if not operation_id.startswith(prefix):
        raise ValueError("The pending action belongs to another task")
    saved = store.get(operation_id)
    if not saved or (saved[1] or {}).get("status") != "needs_approval":
        raise ValueError("This run is not awaiting approval")
    if job.status != "active" or job.policy.expired:
        raise ValueError("Enable the task and check its deadline before continuing")
    fingerprint = task_fingerprint(job)
    if saved[0] != fingerprint:
        raise ValueError("The task changed while awaiting approval; dismiss the old action first")
    previous = saved[1]
    if previous.get("execution_unknown"):
        raise RecoveryBlocked("Inspect the earlier execution's effects before continuing")
    return previous


async def inspect_approval(job, store, operation_id: str) -> dict:
    previous = pending_run(job, store, operation_id)
    current = await asyncio.to_thread(observe, job.policy) if job.policy.input_paths or job.policy.output_paths else None
    delta = changes(current, previous)
    changed = current is not None and current != previous.get("observations")
    complete = current is None or bool(current.get("complete") and (previous.get("observations") or {}).get("complete"))
    output_changed = bool(delta and any(delta["outputs"].values()))
    refreshable = bool(job.goal and complete and not output_changed)
    reason = ("Tracked files could not be inspected." if not complete else
              "Outputs changed; inspect them before replacing this approval." if output_changed else
              "Inputs changed. Prepare a new action before approving." if changed else "")
    try:
        history = store.history(operation_id)
        await asyncio.to_thread(reconcile, history, refresh_reads=True)
        check_revisions(((previous.get("pending_approval") or {}).get("action") or {}).get("revisions", {}))
        read_paths = await asyncio.to_thread(changed_read_paths, history)
        if read_paths:
            changed = True
            delta = delta or {"inputs": {"added": [], "removed": [], "modified": []}, "outputs": {}}
            root = Path(job.policy.workspace)
            delta["inputs"]["modified"] = sorted(set(delta["inputs"]["modified"]) | {
                str(Path(path).relative_to(root)) if Path(path).is_relative_to(root) else path for path in read_paths})
            if not reason:
                reason = "Inputs changed. Prepare a new action before approving."
    except RecoveryBlocked as exc:
        changed, refreshable, reason = True, False, str(exc)
    return {"changed": changed, "refreshable": refreshable, "reason": reason,
            "changes": {group: {kind: paths[:40] for kind, paths in kinds.items()}
                        for group, kinds in (delta or {}).items()},
            "truncated": bool(delta and any(len(paths) > 40 for kinds in delta.values() for paths in kinds.values()))}


async def prepare_continuation(job, store, operation_id: str, *, refresh: bool = False) -> tuple[str, dict, list[dict]]:
    previous = pending_run(job, store, operation_id)
    pending = previous.get("pending_approval") or {}
    action = pending.get("action")
    if (not isinstance(action, dict) or action.get("version") != 1
            or not isinstance(action.get("tool"), str) or not isinstance(action.get("params"), dict)):
        raise ValueError("No resumable action was saved; dismiss this run and review the task")
    if action["tool"] == "harness_check" or action["tool"].startswith(("browser_", "desktop_")):
        raise ValueError("This action needs a fresh observation; dismiss it and continue interactively")
    if pending.get("environment") != execution_config().model_dump(mode="json"):
        raise ValueError("Execution environment settings changed while approval was pending")
    before = previous.get("observations")
    if job.policy.input_paths or job.policy.output_paths:
        current = await asyncio.to_thread(observe, job.policy)
        if (not before or not before.get("complete") or not current.get("complete")
                or (current["outputs"] != before["outputs"] if refresh else current != before)):
            raise RecoveryBlocked("Tracked files changed while approval was pending; inspect them before requesting approval again")
    if refresh and not job.goal:
        raise ValueError("Only agent tasks can prepare an updated action")
    check_revisions(action.get("revisions", {}))
    history = await asyncio.to_thread(reconcile, store.history(operation_id), refresh_reads=refresh)
    prefix = f"cron:{job.id}:"
    budget_owner(job.policy, store.usage(prefix))
    child_id = prefix + "resume:" + uuid4().hex
    store.continue_approval(f"cron:{job.id}", operation_id, child_id, task_fingerprint(job))
    return child_id, previous, history
