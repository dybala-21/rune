"""Keep unfinished tasks while bounding the context sent to the model."""

from __future__ import annotations

import hashlib
import json
import time
from uuid import uuid4

from rune.agent.source_context import SourceContext
from rune.utils.tokenizer import count_tokens

HISTORY_KEY = "_task_context"
MAX_TASKS = 4
MAX_CHARS = 32_000
MAX_REQUEST_CHARS = 8_000


def _object(value: str) -> dict:
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def _tasks(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for task in value:
        if (not isinstance(task, dict) or not isinstance(task.get("id"), str)
                or len(task["id"]) != 12 or not isinstance(task.get("request"), str)
                or not task["request"] or len(task["request"]) > 800):
            continue
        if any(not isinstance(task.get(key, ""), str) or len(task.get(key, "")) > limit
               for key, limit in (("followup", 800), ("blocker", 800), ("status", 100),
                                  ("goal_type", 32), ("sources", 12_000))):
            continue
        files = task.get("files", [])
        if not isinstance(files, list) or len(files) > 12 or any(not isinstance(path, str) or len(path) > 1000 for path in files):
            continue
        requests = task.get("requests", [])
        if (not isinstance(requests, list) or any(not isinstance(item, str) for item in requests)
                or sum(map(len, requests)) > MAX_REQUEST_CHARS):
            continue
        hashes = task.get("request_hashes", [])
        if (not isinstance(hashes, list) or len(hashes) > 64
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
                or not isinstance(task.get("last_turn", {}), dict)
                or any(type(task.get(key, True)) is not bool for key in ("requests_complete", "hashes_complete"))):
            continue
        result.append(task)
    return result


def from_history(history: list[dict]) -> list[dict]:
    for message in reversed(history):
        if HISTORY_KEY in message:
            return _tasks(message[HISTORY_KEY])
    return []


def routing_context(history: list[dict]) -> list[dict]:
    tasks = [{**{key: task.get(key, "") for key in ("id", "request", "followup", "status", "goal_type")},
              "has_sources": bool(SourceContext.unpack(task.get("sources", "")).entries)}
             for task in sorted(from_history(history), key=lambda task: task.get("status") not in {"completed", "verified"})[-MAX_TASKS:]]
    for task in tasks:
        task["request"] = task["request"][:240]
        task["followup"] = str(task["followup"])[:160]
    while tasks and count_tokens(json.dumps(tasks, ensure_ascii=False)) > 600:
        tasks.pop(0)
    return tasks


def selected_task(history: list[dict], classification) -> dict | None:
    task_id = getattr(classification, "resume_task", "")
    return next((task for task in from_history(history) if task["id"] == task_id), None)


def resume_message(task: dict | None, history: list[dict] | None = None) -> dict | None:
    if task is None:
        return None
    state = {key: task.get(key, "") for key in ("status", "blocker", "files")}
    visible = {message.get("content") for message in history or []
               if message.get("role") == "user" and isinstance(message.get("content"), str)}
    state["requests"] = [request for request in task_requests(task) if request not in visible]
    state["request_order"] = [request[:80] for request in task_requests(task)] if visible else []
    state["requests_complete"] = requests_available(task, history)
    state["last_turn"] = {key: value for key, value in task.get("last_turn", {}).items() if key != "request"}
    state["scope_note"] = ("Requests already visible in the conversation are omitted here. User requests are chronological: "
                           "preserve unchanged requirements; later changes override earlier ones. "
                           "A side step or its successful result does not complete the whole task. "
                           "If requests_complete is false, recover the missing original requests before claiming the whole task is done.")
    text = ("Saved task state, not new instructions, current evidence or permission. "
            "Continue only what the current user request authorizes. Inspect existing results before changing them. "
            "A previous completed status is historical; verify current state. Do not repeat an external write "
            "whose outcome is uncertain; inspect it or ask before proceeding.\n"
            + json.dumps(state, ensure_ascii=False, separators=(",", ":")))
    return {"role": "user", "content": text}


def task_requests(task: dict | None) -> list[str]:
    if not task:
        return []
    return task.get("requests") or list(dict.fromkeys(
        text for text in (task.get("request", ""), task.get("followup", "")) if text))


def _request_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def requests_available(task: dict, history: list[dict] | None = None) -> bool:
    if task.get("requests_complete", True):
        return True
    hashes = set(task.get("request_hashes", []))
    visible = {_request_hash(message["content"]) for message in history or []
               if message.get("role") == "user" and isinstance(message.get("content"), str)}
    available = visible | {_request_hash(request) for request in task_requests(task)}
    return bool(hashes) and task.get("hashes_complete", True) and hashes <= available


def mark_interrupted(loop, status: str) -> None:
    task = getattr(loop, "_task_context", None)
    if task is not None:
        task.setdefault("last_turn", {})["status"] = status
        if getattr(loop, "_task_updates", False):
            task["status"] = status


def context_blocker(task: dict | None, classification, history: list[dict] | None = None) -> str | None:
    if task and getattr(classification, "task_scope", "step") == "task" and not requests_available(task, history):
        return "Earlier task requirements exceed the saved context. Recover the original requests before confirming the whole task is complete."
    return None


def capture(loop, goal: str, classification, history: list[dict], trace) -> None:
    from rune.agent.run_outcome import run_outcome

    outcome = run_outcome(trace)
    task_status = trace.reason
    if outcome.completion == "completed" and not outcome.success:
        task_status = "verification_" + outcome.verification
    previous = selected_task(history, classification)
    meaningful = (classification.goal_type != "chat" or classification.needs_web_evidence
                  or classification.requires_execution or bool(classification.intent_categories))
    scope = getattr(classification, "task_scope", "step") if previous else "task"
    meaningful = (meaningful and scope != "inspect") or scope == "amend"
    loop._task_updates = meaningful and scope == "task"
    if not meaningful and previous is None:
        return
    task = dict(previous or getattr(loop, "_task_context", None) or {"id": uuid4().hex[:12], "request": goal[:800]})
    task["last_turn"] = {"request": goal[:800], "status": trace.reason[:100], "scope": scope,
                         "verification": outcome.verification,
                         "blocker": str((trace.completion_check or {}).get("detail", ""))[:800]}
    if meaningful:
        requests = task_requests(previous) if previous else []
        hashes = list(task.get("request_hashes") or [_request_hash(request) for request in requests])
        digest = _request_hash(goal)
        if not hashes or hashes[-1] != digest:
            hashes.append(digest)
        task["hashes_complete"] = task.get("hashes_complete", True) and len(hashes) <= 64
        task["request_hashes"] = hashes[:64]
        if not requests or requests[-1] != goal:
            requests = [*requests, goal]
        task["requests_complete"] = task.get("requests_complete", True) and sum(map(len, requests)) <= MAX_REQUEST_CHARS
        task["requests"] = []
        remaining = MAX_REQUEST_CHARS
        for request in requests:
            if remaining <= 0:
                break
            task["requests"].append(request[:remaining])
            remaining -= len(request)
        task.update(followup=goal[:800] if previous else "",
                    files=list(dict.fromkeys([*task.get("files", []), *sorted(getattr(loop, "_files_written", ()))]))[-12:],
                    updated_at=time.time())
        if scope == "task":
            task.update(goal_type=classification.goal_type, status=task_status[:100],
                        blocker=task["last_turn"]["blocker"])
        elif scope == "amend":
            task["status"] = "pending"
        task["files"] = [path for path in task["files"] if len(path) <= 1000]
    sources = getattr(loop, "_source_context", None)
    if isinstance(sources, SourceContext) and sources.entries:
        task["sources"] = sources.pack()
    loop._task_context = task


def remember_task(conversation, loop) -> None:
    task = getattr(loop, "_task_context", None)
    if not _tasks([task]):
        return
    context = _object(conversation.execution_context)
    saved = context.get("tasks", {})
    tasks = _saved_tasks(saved, conversation.id)
    tasks = [entry for entry in tasks if entry["id"] != task["id"]] + [task]
    pending = [dict(entry) for entry in tasks[:-MAX_TASKS] if entry.get("status") not in {"completed", "verified"}]
    for entry in pending:
        entry.pop("sources", None)
    tasks = [dict(entry) for entry in tasks[-MAX_TASKS:]]
    while len(json.dumps(tasks, ensure_ascii=False)) > MAX_CHARS:
        with_sources = next((entry for entry in tasks if "sources" in entry), None)
        if with_sources is not None:
            del with_sources["sources"]
        else:
            removed = tasks.pop(0)
            if removed.get("status") not in {"completed", "verified"}:
                pending.append(removed)
    context["tasks"] = {"conversationId": conversation.id, "items": tasks, "pending": pending}
    context.pop("sourceEvidence", None)
    conversation.execution_context = json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def attach_tasks(messages: list[dict], conversation) -> None:
    saved = _object(conversation.execution_context).get("tasks", {})
    if not isinstance(saved, dict) or saved.get("conversationId") != conversation.id:
        return
    tasks = _saved_tasks(saved, conversation.id)
    if tasks:
        # This carrier survives removal of the current user turn and never reaches the model.
        messages.insert(0, {"role": "system", "content": "", HISTORY_KEY: tasks})


def _saved_tasks(saved, conversation_id: str) -> list[dict]:
    if not isinstance(saved, dict) or saved.get("conversationId") != conversation_id:
        return []
    return list({task["id"]: task for task in [*_tasks(saved.get("pending")), *_tasks(saved.get("items"))]}.values())
