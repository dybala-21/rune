"""Explicit continuation of interrupted web executions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from rune.agent.execution_journal import RecoveryBlocked, reconcile
from rune.api.run_snapshot import RunSnapshots
from rune.api.run_store import RunStore
from rune.types import CapabilityResult


class ResumeRequest(BaseModel):
    run_id: str = Field(alias="runId", min_length=1, max_length=128)


class RunRecovery:
    def __init__(self, runs: RunSnapshots, store: RunStore, *, browser_for_session=None) -> None:
        self.runs, self.store = runs, store
        self.browser_for_session = browser_for_session

    def workspace_available(self, run_id: str, workspace: str, *, resuming: bool) -> None:
        root = Path(workspace).resolve()
        for other_id in self.store.active_ids():
            if other_id == run_id:
                continue
            other = self.runs.get(other_id)
            if not other or not other.get("workspace"):
                continue
            if not resuming and not other.get("parentRunId"):
                continue
            occupied = Path(other["workspace"]).resolve()
            if occupied.is_relative_to(root) or root.is_relative_to(occupied):
                raise RecoveryBlocked("Another execution is using this workspace. Wait for it to finish before continuing.")

    def records(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        chain = []
        seen = set()
        current = run
        while current:
            if current["runId"] in seen or len(seen) >= 100:
                raise RecoveryBlocked("Execution ancestry is invalid or too long to recover.")
            seen.add(current["runId"])
            if current.get("recoveryVersion") != 1:
                raise RecoveryBlocked("This execution predates durable tool recording and cannot be resumed safely.")
            chain.append(current)
            parent = current.get("parentRunId")
            current = self.runs.get(parent) if parent else None
            if parent and current is None:
                raise RecoveryBlocked("A predecessor execution is missing.")
        records = []
        for item in reversed(chain):
            attempts = self.store.attempts(item["runId"])
            for attempt in attempts:
                if attempt["tool"] == "ask_user" and attempt["state"] != "done":
                    for interaction in item.get("interactions", []):
                        if (interaction["kind"] == "question" and interaction["status"] == "answered"
                                and interaction["request"].get("callId") == attempt["call_id"]):
                            from rune.capabilities.ask_user import AskUserParams, user_response
                            response = interaction["response"]
                            answer = user_response(AskUserParams.model_validate(attempt["params"]),
                                                   response["answer"], response.get("selectedIndex"))
                            attempt.update(state="done", result=asdict(CapabilityResult(
                                success=True, output=f'User responded: "{answer.answer}"',
                                metadata={"reconciled": True},
                            )))
                            break
            records.extend(attempts)
        browser = self.browser_for_session(run["sessionId"]) if self.browser_for_session else None
        return reconcile(records, browser=browser)

    def begin(self, run_id: str) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
        run = self.runs.get(run_id)
        if run is None:
            raise RecoveryBlocked("Execution was not found.")
        child_id = self.store.resumed_child(run_id)
        if child_id:
            child = self.runs.get(child_id)
            if child is None:
                raise RecoveryBlocked("The continuation record is missing.")
            return child, None, []
        if run["status"] != "interrupted":
            raise RecoveryBlocked("Only interrupted executions can be resumed.")
        if self.store.latest_id(run["sessionId"]) != run_id:
            raise RecoveryBlocked("This conversation has newer work. Continue its latest execution instead.")
        workspace = run.get("workspace", "")
        if not workspace or not Path(workspace).is_dir():
            raise RecoveryBlocked("The original workspace is unavailable.")
        self.workspace_available(run_id, workspace, resuming=True)
        records = self.records(run)
        child_id = uuid4().hex[:16]
        self.runs.start(child_id, run["sessionId"], run["goal"], parent_id=run_id)
        child = self.runs.get(child_id)
        assert child is not None
        return child, run, records
