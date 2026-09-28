"""Completion evidence for browser interaction and calculated answers."""

from dataclasses import dataclass

from rune.agent.intent_engine import IntentContract
from rune.types import CapabilityResult


@dataclass
class TaskEvidence:
    browser_actions: int = 0
    browser_observed: bool = False
    browser_failure: str = ""
    computations: int = 0

    @classmethod
    def for_request(cls, goal: str, classification) -> "TaskEvidence":
        from rune.agent.calculation import calculation_context

        # The exact arithmetic supplied in the prompt is already computed evidence.
        return cls(computations=int(bool(calculation_context(goal, classification))))

    def observe(self, name: str, result: CapabilityResult) -> None:
        meta = result.metadata or {}
        if meta.get("cached") or meta.get("replayed"):
            return
        if name in {"browser_batch", "browser_workflow"}:
            for step in meta.get("browser_steps", []):
                self.observe(step["tool"], CapabilityResult(
                    success=step["success"], metadata=step.get("metadata", {}), error=step.get("error"),
                ))
            if not result.success:
                self.browser_failure = result.error or "A browser step did not finish."
        elif name == "browser_act":
            if result.success and meta.get("action_status") == "dispatched":
                self.browser_actions += 1
                self.browser_observed = not meta.get("observation_failed", False)
                if meta.get("action") != "scroll":
                    self.browser_failure = ""
            else:
                self.browser_failure = result.error or "The browser action did not finish."
        elif name in {"browser_observe", "browser_extract"}:
            if self.browser_actions:
                self.browser_observed = result.success
        elif name in {"file_read", "document_read"} and result.success:
            profile = meta.get("table_profile", {})
            if profile.get("complete") and isinstance(profile.get("rows"), int):
                self.computations += 1

    def blocker(self, contract: IntentContract, executions: int) -> str | None:
        if contract.kind == "browser_write":
            if self.browser_failure:
                return ("The requested browser interaction is unfinished: " + self.browser_failure[:500]
                        + " Inspect the page and complete the remaining action, or report the blocker. "
                        "Do not substitute a calculated answer for a requested page update.")
            if not self.browser_actions:
                return "No browser input confirms the requested interaction. Observing the page alone does not complete it."
            if not self.browser_observed:
                return "Read the page after the dispatched action to verify its result. Do not repeat the action."
        if contract.kind == "calculation" and not (self.computations or executions):
            return ("No computed result supports this calculation yet. Read the CSV with file_read for its "
                    "whole-file counts and sums, or calculate the requested result with code. Do not create an unrequested deliverable.")
        return None
