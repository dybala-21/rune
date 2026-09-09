"""Run state projected from committed execution events."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict, deque
from typing import Any

from rune.api.questions import InteractionResponses
from rune.api.run_store import RunStore

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _terminal_status(event: str, data: dict[str, Any]) -> str | None:
    if event == "agent_complete" and (data.get("trust") or {}).get("completionStatus") == "failed":
        return "failed"
    return {"agent_complete": "completed", "agent_error": "failed",
            "agent_aborted": "cancelled", "agent_interrupted": "interrupted"}.get(event)


def _next_status(run: dict[str, Any], event: str, data: dict[str, Any]) -> str:
    terminal = _terminal_status(event, data)
    if terminal:
        return terminal
    if run["status"] in TERMINAL:
        return run["status"]
    if event == "agent_start":
        return "running"
    if event in {"question", "approval_request"} and not (data.get("autonomous") or data.get("autoApproved")):
        return "waiting_input" if event == "question" else "waiting_approval"
    if event in {"question_closed", "approval_closed"}:
        for field, status in (("approval", "waiting_approval"), ("question", "waiting_input")):
            if run[field] and not (event == f"{field}_closed" and run[field]["id"] == data.get("id")):
                return status
        return "running"
    return run["status"]


class RunSnapshots:
    def __init__(self, store: RunStore | None = None) -> None:
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._store = store
        self._responses = InteractionResponses()

    def open(self) -> None:
        if self._store is not None and self._store.open():
            try:
                for run_id in self._store.active_ids():
                    self._get(run_id)
                    self.record("agent_interrupted", {"runId": run_id, "interruptionReason": "server_restart"})
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._runs.clear()

    def interrupt_active(self, reason: str) -> None:
        for run_id, run in list(self._runs.items()):
            if run["status"] not in TERMINAL:
                self.record("agent_interrupted", {"runId": run_id, "interruptionReason": reason})

    def _get(self, run_id: str) -> dict[str, Any] | None:
        if run_id not in self._runs and self._store is not None:
            saved = self._store.load(run_id)
            if saved is not None:
                run, events = saved
                text = run.pop("text", "")
                run.update(_textParts=deque([text]) if text else deque(), _textSize=len(text))
                for event, data, timestamp in events:
                    self._apply(run, event, data, timestamp)
                self._runs[run_id] = run
        return self._runs.get(run_id)

    def start(self, run_id: str, session_id: str, goal: str, *, parent_id: str | None = None) -> None:
        self.open()
        existing = self._get(run_id)
        if existing is not None:
            if existing["sessionId"] != session_id:
                self.record("run_context", {"runId": run_id, "sessionId": session_id})
            return
        self._prune()
        run = {
            "runId": run_id, "sessionId": session_id, "goal": goal,
            "status": "queued", "seq": 0, "startedAt": time.time() * 1000,
            "_textParts": deque(), "_textSize": 0, "toolCalls": [], "stepNumber": 0,
            "question": None, "approval": None, "trust": None,
        }
        if parent_id:
            run["parentRunId"] = parent_id
            parent = self.get(parent_id)
            if parent is None:
                raise ValueError("Predecessor execution was not found")
            for key in ("workspace", "recoveryVersion", "execution", "fileChanges"):
                if key in parent:
                    run[key] = copy.deepcopy(parent[key])
        if self._store is not None:
            if parent_id:
                self._store.create_resumption(parent_id, self._snapshot(run))
            else:
                self._store.create(self._snapshot(run))
        self._runs[run_id] = run

    def record(
        self, event: str, data: dict[str, Any], *,
        response: tuple[str, str, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        self.open()
        run_id = data.get("runId", "")
        run = self._get(run_id) if run_id else None
        if run is None:
            return data
        if run["status"] in TERMINAL and not (
            event in {"question_closed", "approval_closed"}
            or event == "agent_aborted" and run["status"] == "cancelled"
        ):
            return None
        now = time.time() * 1000
        if self._store is not None:
            # Store deltas between checkpoints; copying the full transcript on
            # every token makes long responses progressively slower.
            checkpoint = self._snapshot(run) if run["seq"] and (
                run["seq"] % 128 == 0 or event in {"agent_complete", "agent_error", "agent_aborted", "agent_interrupted"}
            ) else None
            self._store.append(run, event, data, now, checkpoint=checkpoint, response=response,
                               status=_next_status(run, event, data))
        self._apply(run, event, data, now)
        return {**data, "seq": run["seq"], "sessionId": run["sessionId"]}

    @staticmethod
    def _apply(run: dict[str, Any], event: str, data: dict[str, Any], now: float) -> None:
        run["seq"] += 1
        run["status"] = _next_status(run, event, data)
        run["updatedAt"] = now
        if event == "run_context":
            for key in ("sessionId", "workspace", "recoveryVersion", "execution"):
                if key in data:
                    run[key] = data[key]
        elif event == "text_delta":
            if not run["_textSize"]:
                run["textStartedAt"] = now
            if "delta" not in data:
                run["_textParts"].clear()
                run["_textSize"] = 0
            chunk = data.get("delta", data.get("text", ""))[-1_000_000:]
            run["_textParts"].append(chunk)
            run["_textSize"] += len(chunk)
            while run["_textSize"] > 1_000_000:
                run["_textSize"] -= len(run["_textParts"].popleft())
        elif event == "step_start":
            run["stepNumber"] = data.get("stepNumber", 0)
        elif event == "tool_call":
            run["toolCalls"].append({**copy.deepcopy(data), "timestamp": now,
                                     "step": run["stepNumber"]})
            run["toolCalls"] = run["toolCalls"][-3000:]
        elif event == "tool_result":
            if data.get("fileChange"):
                change = data["fileChange"]
                changes = {item["id"]: item for item in run.get("fileChanges", [])}
                changes[change["id"]] = copy.deepcopy(change)
                run["fileChanges"] = list(changes.values())[-100:]
            if "artifactReceipts" in data:
                run["artifactReceipts"] = copy.deepcopy(data["artifactReceipts"])
            for call in run["toolCalls"]:
                if ((data.get("callId") and call.get("callId") == data["callId"])
                    or (not data.get("callId") and call["toolName"] == data.get("toolName")
                        and "result" not in call)):
                    call.update(result=data.get("result", ""), success=data.get("success"),
                                completedAt=now, durationMs=now - call["timestamp"])
                    break
        elif event in {"question", "approval_request"}:
            if data.get("autonomous") or data.get("autoApproved"):
                return
            field = "question" if event == "question" else "approval"
            run[field] = copy.deepcopy(data)
        elif event in {"question_closed", "approval_closed"}:
            field = "question" if event == "question_closed" else "approval"
            if run[field] and run[field]["id"] == data.get("id"):
                run[field] = None
        elif event in {"agent_complete", "agent_error", "agent_aborted", "agent_interrupted"}:
            run["question"] = run["approval"] = None
            for key in ("trust", "answer", "error", "durationMs", "success", "interruptionReason"):
                if key in data:
                    run[key] = copy.deepcopy(data[key])
            if event == "agent_interrupted":
                run["success"] = False
                run["trust"] = {
                    **(run.get("trust") or {}), "completionStatus": "interrupted",
                    "verified": False, "reason": run.get("interruptionReason", "server_restart"),
                    "artifactReceipts": run.get("artifactReceipts", []),
                }

    def replay(self, interaction_id: str, response_id: str, payload: dict[str, Any]) -> bool:
        self.open()
        if self._store is not None:
            return self._store.replay(interaction_id, response_id, payload)
        return self._responses.replay(interaction_id, response_id, payload)

    def accept(self, interaction_id: str, response_id: str, payload: dict[str, Any]) -> None:
        for run in self._runs.values():
            for field in ("question", "approval"):
                if run[field] and run[field]["id"] == interaction_id:
                    if run[field].get("expiresAt", float("inf")) <= time.time() * 1000:
                        raise ValueError("Interaction has expired")
                    self.record(f"{field}_closed", {"runId": run["runId"], "id": interaction_id},
                                response=(interaction_id, response_id, payload))
                    self._responses.remember(interaction_id, response_id, payload)
                    return
        raise ValueError("Interaction is no longer pending")

    @staticmethod
    def _snapshot(run: dict[str, Any]) -> dict[str, Any]:
        snapshot = {key: copy.deepcopy(value) for key, value in run.items() if not key.startswith("_")}
        snapshot["text"] = "".join(run["_textParts"])
        return snapshot

    def latest(self, session_id: str) -> dict[str, Any] | None:
        self.open()
        self._prune()
        if self._store is not None:
            run_id = self._store.latest_id(session_id)
            return self.get(run_id) if run_id else None
        for run in reversed(self._runs.values()):
            if run["sessionId"] == session_id:
                return self._snapshot(run)
        return None

    def get(self, run_id: str) -> dict[str, Any] | None:
        self.open()
        run = self._get(run_id)
        if run is None:
            return None
        snapshot = self._snapshot(run)
        if self._store is not None:
            snapshot["interactions"] = self._store.interactions(run_id)
            # A crash can occur after the tool receipt commits but before its
            # UI event. Read the saved patch without touching today's file.
            changes = {item["id"]: item for item in snapshot.get("fileChanges", [])}
            for attempt in self._store.attempts(run_id):
                if attempt["tool"] not in {"file_write", "file_edit", "file_delete"}:
                    continue
                change = ((attempt.get("result") or {}).get("metadata") or {}).get("fileChange")
                if change and attempt["state"] == "done":
                    changes[change["id"]] = change
            snapshot["fileChanges"] = list(changes.values())[-100:]
        return snapshot

    def _prune(self) -> None:
        finished = [key for key, run in self._runs.items() if run["status"] in TERMINAL]
        cutoff = time.time() * 1000 - 3_600_000
        for key in finished:
            if key in finished[:-50] or self._runs[key]["updatedAt"] < cutoff:
                del self._runs[key]
