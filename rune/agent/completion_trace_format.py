"""Completion trace formatter - format traces for display and logging.

Ported from src/agent/completion-trace-format.ts (179 lines) - converts
completion traces into user-facing summary strings.

formatCompletionTraceForUser(): User-visible trace summary.
summarizeCompletionTrace(): Progress summary (done/total/blocked).
formatFinalAgentResponseForUser(): Final response formatter for TUI/Gateway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

@dataclass(slots=True)
class CompletionRequirementTrace:
    """A single requirement trace item for display."""

    id: str = ""
    description: str = ""
    required: bool = False
    status: str = ""  # "done" | "blocked" | "skipped"
    failure_reason: str = ""


@dataclass(slots=True)
class CompletionEvidenceTrace:
    """Evidence counts for trace display."""

    reads: int = 0
    writes: int = 0
    executions: int = 0
    verifications: int = 0
    browser_reads: int = 0
    browser_writes: int = 0
    changed_files: int = 0


@dataclass(slots=True)
class CompletionContractTrace:
    """Contract info for trace display."""

    kind: str = ""
    tool_requirement: str = ""
    grounding_requirement: str = ""
    resolved: bool = False
    source: str = ""
    unresolved_reason: str = ""


@dataclass(slots=True)
class CompletionPlanTrace:
    """Plan info for trace display."""

    action_plan: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    verification_candidates: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompletionTraceDisplay:
    """Full completion trace for display purposes."""

    outcome: str = ""  # "verified" | "partial" | "blocked"
    requirements: list[CompletionRequirementTrace] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    evidence: CompletionEvidenceTrace | None = None
    contract: CompletionContractTrace | None = None
    contract_plan: CompletionPlanTrace | None = None
    workspace_root: str = ""
    primary_execution_root: str = ""
    workspace_warning: str = ""


@dataclass(slots=True)
class CompletionProgressSummary:
    """Progress summary for display."""

    required_done: int = 0
    required_total: int = 0
    blocked: list[CompletionRequirementTrace] = field(default_factory=list)


@dataclass(slots=True)
class FinalAgentResponseInput:
    """Input for formatting the final agent response."""

    success: bool
    answer: str = ""
    error: str = ""
    completion_trace: CompletionTraceDisplay | None = None
    compaction_notices: list[str] = field(default_factory=list)
    show_compaction_notice: bool = False


# Helpers

def _has_outstanding_requirements(trace: CompletionTraceDisplay | None) -> bool:
    """Check if there are outstanding required but blocked items."""
    if not trace:
        return False
    if trace.outcome in ("partial", "blocked"):
        return True
    return any(r.required and r.status == "blocked" for r in trace.requirements)


def _format_outcome_line(trace: CompletionTraceDisplay | None) -> str | None:
    if not trace or not trace.outcome:
        return None
    if trace.outcome == "verified":
        return "Verification status: verified"
    if trace.outcome == "partial":
        return "Verification status: partial (outstanding items)"
    return "Verification status: blocked"


def _format_workspace_line(trace: CompletionTraceDisplay | None) -> str | None:
    effective_root = trace.primary_execution_root or trace.workspace_root if trace else ""
    if not effective_root:
        return None
    return f"Workspace: {effective_root}"


def _format_workspace_warning_line(trace: CompletionTraceDisplay | None) -> str | None:
    if not trace or not trace.workspace_warning:
        return None
    return f"Path warning: {trace.workspace_warning}"


def _format_evidence_line(
    trace: CompletionTraceDisplay | None,
    has_outstanding: bool,
) -> str | None:
    if not trace or not trace.evidence or not has_outstanding:
        return None
    ev = trace.evidence
    reads = ev.reads + ev.browser_reads
    writes = ev.writes + ev.browser_writes
    return (
        f"Execution evidence: read={reads} write={writes} "
        f"exec={ev.executions} verify={ev.verifications} "
        f"changed={ev.changed_files}"
    )


def _format_contract_line(trace: CompletionTraceDisplay | None) -> str | None:
    if not trace or not trace.contract:
        return None
    c = trace.contract
    resolution = (
        f"resolved:{c.source}" if c.resolved
        else f"unresolved:{c.unresolved_reason or 'unknown'}:{c.source}"
    )
    return (
        f"Contract: {c.kind} | tool={c.tool_requirement} | "
        f"grounding={c.grounding_requirement} | {resolution}"
    )


def _format_plan_line(trace: CompletionTraceDisplay | None) -> str | None:
    if not trace or not trace.contract_plan:
        return None
    plan = trace.contract_plan
    top_verifications = ", ".join(plan.verification_candidates[:2])
    return (
        f"Plan: {len(plan.action_plan)} steps | "
        f"{len(plan.completion_criteria)} criteria | "
        f"verification: {top_verifications}"
    )


# Public API

def summarize_completion_trace(
    trace: CompletionTraceDisplay | None,
) -> CompletionProgressSummary | None:
    """Summarize completion trace into a progress report."""
    if not trace:
        return None
    required = [r for r in trace.requirements if r.required]
    if not required:
        return None

    informative = [r for r in required if r.id != "hard_failure_signals"]
    target = informative if informative else required
    target_blocked = [r for r in target if r.status == "blocked"]

    if (
        not informative
        and not target_blocked
        and not trace.hard_failures
    ):
        return None

    required_done = sum(1 for r in target if r.status == "done")
    return CompletionProgressSummary(
        required_done=required_done,
        required_total=len(target),
        blocked=target_blocked,
    )


def format_completion_trace_for_user(
    trace: CompletionTraceDisplay | None,
    *,
    include_blocked_details: bool = False,
) -> str | None:
    """Convert completion trace to a user-facing summary string.

    Note: numeric fulfillment ratios are NOT shown - only actionable
    information like contract/plan/outstanding details are exposed.
    """
    summary = summarize_completion_trace(trace)
    if not summary:
        return None

    has_outstanding = _has_outstanding_requirements(trace)
    include_planning_context = (
        (trace is not None and trace.outcome != "verified") or has_outstanding
    )
    is_verified = trace is not None and trace.outcome == "verified"

    lines: list[str] = []

    if not is_verified:
        outcome_line = _format_outcome_line(trace)
        if outcome_line:
            lines.append(outcome_line)

    if not is_verified:
        workspace_line = _format_workspace_line(trace)
        if workspace_line:
            lines.append(workspace_line)

    warning_line = _format_workspace_warning_line(trace)
    if warning_line:
        lines.append(warning_line)

    if include_planning_context:
        contract_line = _format_contract_line(trace)
        if contract_line:
            lines.append(contract_line)
        plan_line = _format_plan_line(trace)
        if plan_line:
            lines.append(plan_line)

    evidence_line = _format_evidence_line(trace, has_outstanding)
    if evidence_line:
        lines.append(evidence_line)

    if include_blocked_details and summary.blocked:
        for item in summary.blocked[:3]:
            detail = item.failure_reason or item.description
            lines.append(f"- Outstanding: {detail}")

    if not lines:
        return None
    return "\n".join(lines)


DEFAULT_SUCCESS_MESSAGE = "Task completed."


def format_final_agent_response_for_user(inp: FinalAgentResponseInput) -> str:
    """Format the final agent response for user display.

    success=true: show agent response only (no metadata noise)
    success=false: show agent response only (LLM continuation handles explanations)
    """
    raw_answer = (inp.answer or inp.error or DEFAULT_SUCCESS_MESSAGE).strip()
    answer = raw_answer if raw_answer else DEFAULT_SUCCESS_MESSAGE

    compaction_prefix = ""
    if (
        inp.show_compaction_notice
        and inp.compaction_notices
    ):
        compaction_prefix = (
            f"Context optimization notice: {' / '.join(inp.compaction_notices)}\n\n"
        )

    return f"{compaction_prefix}{answer}"
