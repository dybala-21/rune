"""Exercise web submissions with real models and isolated conversation state."""

import json
import time
from pathlib import Path
from uuid import uuid4

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


class Workflow:
    def __init__(self, client, workspace, provider, model):
        self.client, self.workspace = client, workspace
        self.provider, self.model = provider, model
        self.runs = []
        self.decisions = []
        self.session = uuid4().hex

    def submit(self, goal, attachments=None, *, editable_files=(), allow_new_tests=False, deny_approvals=False):
        started = time.monotonic()
        payload = {"text": goal, "sessionId": self.session, "requestId": uuid4().hex, "attachments": attachments}
        response = self.client.post("/api/message", json=payload)
        assert response.status_code == 200, response.text
        accepted = response.json()
        first_text = None
        approvals = []
        denials = []
        answered = set()
        while time.monotonic() - started < 210:
            run = self.client.get("/api/runs/snapshot", params={"sessionId": self.session}).json()["run"]
            if run and run.get("text") and first_text is None:
                first_text = time.monotonic() - started
            if run and run["status"] in TERMINAL:
                break
            approval = run.get("approval") if run else None
            if approval and approval["id"] in answered:
                time.sleep(.1)
                continue
            if approval and deny_approvals:
                reply = self.client.post("/api/approval", json={
                    "id": approval["id"], "decision": "deny", "responseId": uuid4().hex,
                })
                assert reply.status_code == 200, reply.text
                denials.append(approval)
                answered.add(approval["id"])
                continue
            if approval and self._can_approve_edit(run, editable_files, allow_new_tests=allow_new_tests):
                reply = self.client.post("/api/approval", json={
                    "id": approval["id"], "decision": "approve_once", "responseId": uuid4().hex,
                })
                assert reply.status_code == 200, reply.text
                approvals.append(approval)
                answered.add(approval["id"])
                continue
            if run and (run.get("approval") or run.get("question")):
                self.client.post("/api/abort", json={"runId": accepted["runId"]})
                break
            time.sleep(.1)
        else:
            self.client.post("/api/abort", json={"runId": accepted["runId"]})
        run = self.client.get("/api/runs/snapshot", params={"sessionId": self.session}).json()["run"]
        assert run["runId"] == accepted["runId"]
        self.runs.append({"goal": goal, "seconds": round(time.monotonic() - started, 3),
                          "first_text_seconds": first_text, "acceptance": accepted,
                          "approved_fixture_edits": approvals, "denied_fixture_actions": denials, "snapshot": run})
        # The terminal event can arrive before server cleanup finishes.
        for _ in range(100):
            active = self.client.post("/api/v1/rpc", json={"method": "runs.active", "params": {}}).json()["data"]["runIds"]
            if accepted["runId"] not in active:
                break
            time.sleep(.05)
        return run

    def _can_approve_edit(self, run, editable_files, *, allow_new_tests=False):
        calls = run.get("toolCalls") or []
        if not calls or calls[-1]["toolName"] not in {"file_edit", "file_write"}:
            return False
        if run["approval"].get("command") != calls[-1]["toolName"]:
            return False
        target = Path(calls[-1]["args"].get("path", "")).resolve()
        named = target in {(self.workspace / name).resolve() for name in editable_files}
        new_test = (allow_new_tests and calls[-1]["toolName"] == "file_write"
                    and target.parent == self.workspace and not target.exists()
                    and target.name.startswith("test_") and target.suffix == ".py")
        return target.is_relative_to(self.workspace) and (named or new_test)

    def write_report(self, path: Path, outcome: str):
        from scripts.e2e_report import metrics

        artifacts = {p.name: p.read_text() for p in self.workspace.iterdir()
                     if p.is_file() and p.suffix in {".py", ".csv", ".txt"} and p.stat().st_size < 100_000}
        report = {"provider": self.provider, "model": self.model, "outcome": outcome, "case": path.stem,
                  "decision_backend": getattr(self, "decision_backend", None),
                  "scope": "Web API, real model, managed browser. Background learning disabled; no cross-model fallback.",
                  "runs": self.runs, "decisions": self.decisions, "artifacts": artifacts,
                  "metrics": metrics(self.runs)}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        costs = [(r["snapshot"].get("usage") or {}).get("cost", {}).get("usd") for r in self.runs]
        print(json.dumps({"provider": self.provider, "case": path.stem, "outcome": outcome,
                          "seconds": sum(r["seconds"] for r in self.runs),
                          "usd": sum(costs) if costs and all(c is not None for c in costs) else None}), flush=True)


def gui_calls(run):
    return [call for call in run["toolCalls"] if call["toolName"].startswith(("browser", "desktop", "computer"))]
