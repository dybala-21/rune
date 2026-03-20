"""Completion gate for RUNE - verifies task completion before finishing.

Ported from src/agent/completion-gate.ts (614 lines) - evaluates 18
requirements to determine if a task is truly complete, partially done,
or blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Type aliases

ToolRequirement = Literal["none", "read", "write"]
OutputExpectation = Literal["text", "file", "either"]

# Evidence data classes


@dataclass(slots=True)
class ExecutionEvidenceSnapshot:
    """Numeric counts of tool usage evidence."""
    reads: int = 0
    writes: int = 0
    executions: int = 0
    verifications: int = 0
    browser_reads: int = 0
    browser_writes: int = 0
    web_searches: int = 0
    web_fetches: int = 0
    file_reads: int = 0
    unique_file_reads: int = 0


@dataclass(slots=True)
class EvidenceSamples:
    """Sample strings for evidence reporting."""
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    executions: list[str] = field(default_factory=list)
    verifications: list[str] = field(default_factory=list)
    browser_reads: list[str] = field(default_factory=list)
    browser_writes: list[str] = field(default_factory=list)
    web_searches: list[str] = field(default_factory=list)
    web_fetches: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ServiceTaskEvidenceSnapshot:
    """Evidence for service-type tasks (start/probe/cleanup)."""
    starts: int = 0
    runtime_probes: int = 0
    cleanups: int = 0


@dataclass(slots=True)
class ServiceTaskEvidenceSamples:
    """Sample strings for service task evidence."""
    starts: list[str] = field(default_factory=list)
    runtime_probes: list[str] = field(default_factory=list)
    cleanups: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RequirementTraceItem:
    """Individual requirement evaluation result."""
    id: str
    description: str
    required: bool
    status: Literal["done", "blocked", "skipped"]
    evidence: str = ""
    failure_reason: str = ""


@dataclass(slots=True)
class WorkspaceAlignmentSnapshot:
    """Workspace path information for alignment checking."""
    workspace_root: str = ""
    requested_workspace_path: str = ""
    primary_execution_root: str = ""
    execution_roots: list[str] = field(default_factory=list)


# Gate Input / Output

@dataclass(slots=True)
class CompletionGateInput:
    """All inputs needed to evaluate the completion gate."""
    # Core requirements
    tool_requirement: ToolRequirement = "none"
    output_expectation: OutputExpectation = "text"
    intent_resolved: bool = False
    requires_code_verification: bool = False
    requires_code_write_artifact: bool = False

    # Evidence
    evidence: ExecutionEvidenceSnapshot = field(default_factory=ExecutionEvidenceSnapshot)
    evidence_samples: EvidenceSamples = field(default_factory=EvidenceSamples)

    # File artifacts
    changed_files_count: int = 0
    structured_write_count: int = 0
    structured_write_samples: list[str] = field(default_factory=list)

    # Service tasks
    service_task: ServiceTaskEvidenceSnapshot | None = None
    service_task_samples: ServiceTaskEvidenceSamples | None = None

    # Workspace
    workspace: WorkspaceAlignmentSnapshot | None = None

    # Hard failures
    hard_failures: list[str] = field(default_factory=list)
    infra_failures: list[str] = field(default_factory=list)

    # Research / analysis requirements
    grounding_requirement: bool = False
    analysis_depth_min_reads: int = 0
    module_count: int = 0
    min_module_coverage: int = 0
    deep_analysis_tools: int = 0
    min_deep_analysis_tools: int = 0
    read_rounds: int = 0
    min_read_rounds: int = 0

    # Web requirements
    min_web_searches: int = 0
    min_web_fetches: int = 0

    # Answer length
    answer_length: int = 0


@dataclass(slots=True)
class CompletionGateResult:
    """Result of the completion gate evaluation."""
    outcome: Literal["verified", "partial", "blocked"]
    success: bool
    error: str | None = None
    hard_failures: list[str] = field(default_factory=list)
    missing_requirement_ids: list[str] = field(default_factory=list)
    requirements: list[RequirementTraceItem] = field(default_factory=list)
    workspace_warning: str | None = None


# Requirement IDs

REQUIREMENT_IDS: dict[str, str] = {
    "INTENT_RESOLVED": "R01_INTENT_RESOLVED",
    "TOOL_USAGE": "R02_TOOL_USAGE",
    "READ_EVIDENCE": "R03_READ_EVIDENCE",
    "WRITE_EVIDENCE": "R04_WRITE_EVIDENCE",
    "EXECUTION_EVIDENCE": "R05_EXECUTION_EVIDENCE",
    "VERIFICATION": "R06_VERIFICATION",
    "FILE_ARTIFACT": "R07_FILE_ARTIFACT",
    "CODE_WRITE_ARTIFACT": "R08_CODE_WRITE_ARTIFACT",
    "SERVICE_START": "R09_SERVICE_START",
    "SERVICE_PROBE": "R10_SERVICE_PROBE",
    "SERVICE_CLEANUP": "R11_SERVICE_CLEANUP",
    "WORKSPACE_ALIGNMENT": "R12_WORKSPACE_ALIGNMENT",
    "NO_HARD_FAILURES": "R13_NO_HARD_FAILURES",
    "GROUNDING": "R14_GROUNDING",
    "ANALYSIS_DEPTH": "R15_ANALYSIS_DEPTH",
    "MODULE_COVERAGE": "R16_MODULE_COVERAGE",
    "DEEP_ANALYSIS": "R17_DEEP_ANALYSIS",
    "WEB_EVIDENCE": "R18_WEB_EVIDENCE",
}


# Helpers

def _clip_list(items: list[str], max_items: int = 5) -> str:
    """Clip a list to *max_items* and format as comma-separated string."""
    if not items:
        return "(none)"
    clipped = items[:max_items]
    suffix = f" (+{len(items) - max_items} more)" if len(items) > max_items else ""
    return ", ".join(clipped) + suffix


def _summarize_hard_failures(failures: list[str]) -> str:
    """Format hard failures for error reporting."""
    if not failures:
        return ""
    return "; ".join(failures[:5])


def _normalize_workspace_path(path: str) -> str:
    """Normalize a workspace path for comparison."""
    import os
    if not path:
        return ""
    return os.path.normpath(os.path.expanduser(path)).rstrip("/")


def _is_same_or_child_path(child: str, parent: str) -> bool:
    """Check if *child* is the same as or under *parent*."""
    if not child or not parent:
        return False
    norm_child = _normalize_workspace_path(child)
    norm_parent = _normalize_workspace_path(parent)
    return norm_child == norm_parent or norm_child.startswith(norm_parent + "/")


def _is_aligned_path(execution_root: str, workspace_root: str) -> bool:
    """Check if the execution root is aligned with the workspace root."""
    return _is_same_or_child_path(execution_root, workspace_root)


def _evaluate_workspace_alignment(
    workspace: WorkspaceAlignmentSnapshot,
) -> tuple[bool, str]:
    """Evaluate workspace alignment. Returns (aligned, warning_message)."""
    if not workspace.workspace_root:
        return True, ""

    # Check primary execution root
    if workspace.primary_execution_root:
        if _is_aligned_path(workspace.primary_execution_root, workspace.workspace_root):
            return True, ""
        return False, (
            f"Primary execution root '{workspace.primary_execution_root}' "
            f"is not under workspace '{workspace.workspace_root}'"
        )

    # Check all execution roots
    if workspace.execution_roots:
        misaligned = [
            root for root in workspace.execution_roots
            if not _is_aligned_path(root, workspace.workspace_root)
        ]
        if misaligned:
            return False, (
                f"Execution roots not under workspace: {', '.join(misaligned[:3])}"
            )

    return True, ""


# Main evaluation

# Module-level cache for incremental evaluation
_last_input: CompletionGateInput | None = None
_last_result: CompletionGateResult | None = None


def evaluate_completion_gate(inp: CompletionGateInput) -> CompletionGateResult:
    """Evaluate all 18 requirements and produce a completion gate result.

    Each requirement is evaluated independently. The overall outcome is:
    - **verified**: all required checks pass
    - **partial**: some non-critical checks fail
    - **blocked**: hard failures or critical checks fail

    Uses equality-based caching: if input hasn't changed since last
    call, returns the cached result (avoids re-evaluating 18 requirements).
    """
    global _last_input, _last_result
    if _last_result is not None and _last_input is not None and inp == _last_input:
        return _last_result
    requirements: list[RequirementTraceItem] = []
    missing: list[str] = []
    ev = inp.evidence
    es = inp.evidence_samples

    # --- R01: Intent Resolved -----------------------------------------------
    r01 = RequirementTraceItem(
        id=REQUIREMENT_IDS["INTENT_RESOLVED"],
        description="User intent is resolved",
        required=True,
        status="done" if inp.intent_resolved else "blocked",
        evidence="Intent marked as resolved" if inp.intent_resolved else "",
        failure_reason="" if inp.intent_resolved else "Intent not yet resolved",
    )
    requirements.append(r01)
    if r01.status != "done":
        missing.append(r01.id)

    # --- R02: Tool Usage ----------------------------------------------------
    needs_tools = inp.tool_requirement != "none"
    has_tool_usage = (ev.reads + ev.writes + ev.executions) > 0
    r02_ok = (not needs_tools) or has_tool_usage
    r02 = RequirementTraceItem(
        id=REQUIREMENT_IDS["TOOL_USAGE"],
        description="Required tool usage observed",
        required=needs_tools,
        status="done" if r02_ok else "blocked",
        evidence=f"reads={ev.reads}, writes={ev.writes}, execs={ev.executions}",
        failure_reason="" if r02_ok else f"Tool requirement '{inp.tool_requirement}' not met",
    )
    requirements.append(r02)
    if r02.required and r02.status != "done":
        missing.append(r02.id)

    # --- R03: Read Evidence -------------------------------------------------
    needs_read = inp.tool_requirement in ("read", "write")
    r03_ok = (not needs_read) or ev.reads > 0 or ev.file_reads > 0
    r03 = RequirementTraceItem(
        id=REQUIREMENT_IDS["READ_EVIDENCE"],
        description="File read evidence",
        required=needs_read,
        status="done" if r03_ok else "blocked",
        evidence=f"reads={ev.reads}, file_reads={ev.file_reads}, samples={_clip_list(es.reads)}",
        failure_reason="" if r03_ok else "No file reads observed",
    )
    requirements.append(r03)
    if r03.required and r03.status != "done":
        missing.append(r03.id)

    # --- R04: Write Evidence ------------------------------------------------
    needs_write = inp.tool_requirement == "write"
    r04_ok = (not needs_write) or ev.writes > 0
    r04 = RequirementTraceItem(
        id=REQUIREMENT_IDS["WRITE_EVIDENCE"],
        description="File write evidence",
        required=needs_write,
        status="done" if r04_ok else "blocked",
        evidence=f"writes={ev.writes}, samples={_clip_list(es.writes)}",
        failure_reason="" if r04_ok else "No file writes observed",
    )
    requirements.append(r04)
    if r04.required and r04.status != "done":
        missing.append(r04.id)

    # --- R05: Execution Evidence --------------------------------------------
    needs_exec = inp.tool_requirement == "write"
    r05_ok = (not needs_exec) or ev.executions > 0
    r05 = RequirementTraceItem(
        id=REQUIREMENT_IDS["EXECUTION_EVIDENCE"],
        description="Command execution evidence",
        required=needs_exec,
        status="done" if r05_ok else "skipped",
        evidence=f"executions={ev.executions}, samples={_clip_list(es.executions)}",
        failure_reason="" if r05_ok else "No command executions observed",
    )
    requirements.append(r05)
    if r05.required and r05.status == "blocked":
        missing.append(r05.id)

    # --- R06: Verification --------------------------------------------------
    r06_ok = (not inp.requires_code_verification) or ev.verifications > 0
    r06 = RequirementTraceItem(
        id=REQUIREMENT_IDS["VERIFICATION"],
        description="Code verification (tests/lint)",
        required=inp.requires_code_verification,
        status="done" if r06_ok else "blocked",
        evidence=f"verifications={ev.verifications}, samples={_clip_list(es.verifications)}",
        failure_reason="" if r06_ok else "Code verification required but not observed",
    )
    requirements.append(r06)
    if r06.required and r06.status != "done":
        missing.append(r06.id)

    # --- R07: File Artifact -------------------------------------------------
    needs_file_artifact = inp.output_expectation in ("file", "either")
    has_file_artifact = inp.changed_files_count > 0 or inp.structured_write_count > 0
    r07_ok = (not needs_file_artifact) or has_file_artifact or inp.output_expectation == "either"
    r07 = RequirementTraceItem(
        id=REQUIREMENT_IDS["FILE_ARTIFACT"],
        description="File artifact produced",
        required=needs_file_artifact and inp.output_expectation == "file",
        status="done" if has_file_artifact else ("skipped" if inp.output_expectation == "either" else "blocked"),
        evidence=(
            f"changed_files={inp.changed_files_count}, "
            f"structured_writes={inp.structured_write_count}, "
            f"samples={_clip_list(inp.structured_write_samples)}"
        ),
        failure_reason="" if r07_ok else "Expected file output but none produced",
    )
    requirements.append(r07)
    if r07.required and r07.status == "blocked":
        missing.append(r07.id)

    # --- R08: Code Write Artifact -------------------------------------------
    r08_ok = (not inp.requires_code_write_artifact) or inp.structured_write_count > 0
    r08 = RequirementTraceItem(
        id=REQUIREMENT_IDS["CODE_WRITE_ARTIFACT"],
        description="Code write artifact produced",
        required=inp.requires_code_write_artifact,
        status="done" if r08_ok else "blocked",
        evidence=f"structured_writes={inp.structured_write_count}",
        failure_reason="" if r08_ok else "Code write artifact required but not produced",
    )
    requirements.append(r08)
    if r08.required and r08.status != "done":
        missing.append(r08.id)

    # --- R09-R11: Service Task Requirements ---------------------------------
    st = inp.service_task
    has_service = st is not None

    r09_ok = (not has_service) or (st is not None and st.starts > 0)
    r09 = RequirementTraceItem(
        id=REQUIREMENT_IDS["SERVICE_START"],
        description="Service started",
        required=has_service,
        status="done" if r09_ok else "blocked",
        evidence=f"starts={st.starts if st else 0}",
        failure_reason="" if r09_ok else "Service not started",
    )
    requirements.append(r09)
    if r09.required and r09.status != "done":
        missing.append(r09.id)

    r10_ok = (not has_service) or (st is not None and st.runtime_probes > 0)
    r10 = RequirementTraceItem(
        id=REQUIREMENT_IDS["SERVICE_PROBE"],
        description="Service runtime probe",
        required=has_service,
        status="done" if r10_ok else "blocked",
        evidence=f"probes={st.runtime_probes if st else 0}",
        failure_reason="" if r10_ok else "No runtime probes observed",
    )
    requirements.append(r10)
    if r10.required and r10.status != "done":
        missing.append(r10.id)

    r11_ok = (not has_service) or (st is not None and st.cleanups > 0)
    r11 = RequirementTraceItem(
        id=REQUIREMENT_IDS["SERVICE_CLEANUP"],
        description="Service cleanup",
        required=has_service,
        status="done" if r11_ok else "skipped",
        evidence=f"cleanups={st.cleanups if st else 0}",
        failure_reason="" if r11_ok else "Service cleanup not observed",
    )
    requirements.append(r11)
    # Service cleanup is non-blocking (skipped, not blocked)

    # --- R12: Workspace Alignment -------------------------------------------
    ws = inp.workspace
    ws_aligned = True
    ws_warning = ""
    if ws is not None:
        ws_aligned, ws_warning = _evaluate_workspace_alignment(ws)

    r12 = RequirementTraceItem(
        id=REQUIREMENT_IDS["WORKSPACE_ALIGNMENT"],
        description="Workspace alignment",
        required=ws is not None,
        status="done" if ws_aligned else "blocked",
        evidence=(
            f"workspace={ws.workspace_root if ws else ''}, "
            f"exec_root={ws.primary_execution_root if ws else ''}"
        ),
        failure_reason=ws_warning,
    )
    requirements.append(r12)
    if r12.required and r12.status != "done":
        missing.append(r12.id)

    # --- R13: No Hard Failures ----------------------------------------------
    r13_ok = len(inp.hard_failures) == 0
    r13 = RequirementTraceItem(
        id=REQUIREMENT_IDS["NO_HARD_FAILURES"],
        description="No hard failures",
        required=True,
        status="done" if r13_ok else "blocked",
        evidence="No hard failures" if r13_ok else _summarize_hard_failures(inp.hard_failures),
        failure_reason="" if r13_ok else f"{len(inp.hard_failures)} hard failure(s)",
    )
    requirements.append(r13)
    if r13.status != "done":
        missing.append(r13.id)

    # --- R14: Grounding Requirement -----------------------------------------
    r14_ok = (not inp.grounding_requirement) or (ev.web_searches > 0 or ev.web_fetches > 0)
    r14 = RequirementTraceItem(
        id=REQUIREMENT_IDS["GROUNDING"],
        description="Grounding via web evidence",
        required=inp.grounding_requirement,
        status="done" if r14_ok else "blocked",
        evidence=f"web_searches={ev.web_searches}, web_fetches={ev.web_fetches}",
        failure_reason="" if r14_ok else "Grounding required but no web evidence",
    )
    requirements.append(r14)
    if r14.required and r14.status != "done":
        missing.append(r14.id)

    # --- R15: Analysis Depth ------------------------------------------------
    min_reads = inp.analysis_depth_min_reads
    r15_required = min_reads > 0
    r15_ok = (not r15_required) or ev.unique_file_reads >= min_reads
    r15 = RequirementTraceItem(
        id=REQUIREMENT_IDS["ANALYSIS_DEPTH"],
        description=f"Analysis depth (min {min_reads} unique reads)",
        required=r15_required,
        status="done" if r15_ok else "blocked",
        evidence=f"unique_file_reads={ev.unique_file_reads}, min={min_reads}",
        failure_reason="" if r15_ok else f"Only {ev.unique_file_reads}/{min_reads} unique file reads",
    )
    requirements.append(r15)
    if r15.required and r15.status != "done":
        missing.append(r15.id)

    # --- R16: Module Coverage -----------------------------------------------
    r16_required = inp.min_module_coverage > 0 and inp.module_count > 0
    coverage = ev.unique_file_reads
    r16_ok = (not r16_required) or coverage >= inp.min_module_coverage
    r16 = RequirementTraceItem(
        id=REQUIREMENT_IDS["MODULE_COVERAGE"],
        description=f"Module coverage ({inp.min_module_coverage}/{inp.module_count})",
        required=r16_required,
        status="done" if r16_ok else "blocked",
        evidence=f"covered={coverage}, min={inp.min_module_coverage}, total={inp.module_count}",
        failure_reason="" if r16_ok else (
            f"Module coverage {coverage}/{inp.min_module_coverage} insufficient"
        ),
    )
    requirements.append(r16)
    if r16.required and r16.status != "done":
        missing.append(r16.id)

    # --- R17: Deep Analysis Tools -------------------------------------------
    r17_required = inp.min_deep_analysis_tools > 0
    r17_ok = (not r17_required) or inp.deep_analysis_tools >= inp.min_deep_analysis_tools
    r17 = RequirementTraceItem(
        id=REQUIREMENT_IDS["DEEP_ANALYSIS"],
        description=f"Deep analysis tools (min {inp.min_deep_analysis_tools})",
        required=r17_required,
        status="done" if r17_ok else "blocked",
        evidence=f"used={inp.deep_analysis_tools}, min={inp.min_deep_analysis_tools}",
        failure_reason="" if r17_ok else (
            f"Deep analysis tools {inp.deep_analysis_tools}/{inp.min_deep_analysis_tools}"
        ),
    )
    requirements.append(r17)
    if r17.required and r17.status != "done":
        missing.append(r17.id)

    # --- R18: Web Evidence --------------------------------------------------
    r18_required = inp.min_web_searches > 0 or inp.min_web_fetches > 0
    r18_ok_search = ev.web_searches >= inp.min_web_searches
    r18_ok_fetch = ev.web_fetches >= inp.min_web_fetches
    r18_ok = (not r18_required) or (r18_ok_search and r18_ok_fetch)
    r18 = RequirementTraceItem(
        id=REQUIREMENT_IDS["WEB_EVIDENCE"],
        description="Web evidence requirement",
        required=r18_required,
        status="done" if r18_ok else "blocked",
        evidence=(
            f"searches={ev.web_searches}/{inp.min_web_searches}, "
            f"fetches={ev.web_fetches}/{inp.min_web_fetches}"
        ),
        failure_reason="" if r18_ok else "Web evidence requirement not met",
    )
    requirements.append(r18)
    if r18.required and r18.status != "done":
        missing.append(r18.id)

    # --- Aggregate outcome --------------------------------------------------

    result: CompletionGateResult

    # Hard failures always block
    if inp.hard_failures:
        result = CompletionGateResult(
            outcome="blocked",
            success=False,
            error=f"Hard failures: {_summarize_hard_failures(inp.hard_failures)}",
            hard_failures=list(inp.hard_failures),
            missing_requirement_ids=missing,
            requirements=requirements,
            workspace_warning=ws_warning or None,
        )
    else:
        # Count blocked required requirements
        blocked_required = [
            r for r in requirements
            if r.required and r.status == "blocked"
        ]

        if not blocked_required:
            result = CompletionGateResult(
                outcome="verified",
                success=True,
                missing_requirement_ids=[],
                requirements=requirements,
                workspace_warning=ws_warning or None,
            )
        else:
            # Check if the only blocked items are non-critical
            critical_ids = {
                REQUIREMENT_IDS["INTENT_RESOLVED"],
                REQUIREMENT_IDS["NO_HARD_FAILURES"],
            }
            blocked_ids = {r.id for r in blocked_required}
            has_critical_block = bool(blocked_ids & critical_ids)

            if has_critical_block:
                result = CompletionGateResult(
                    outcome="blocked",
                    success=False,
                    error=f"Critical requirements not met: {', '.join(r.id for r in blocked_required)}",
                    missing_requirement_ids=missing,
                    requirements=requirements,
                    workspace_warning=ws_warning or None,
                )
            else:
                result = CompletionGateResult(
                    outcome="partial",
                    success=False,
                    error=f"Partial completion: {len(blocked_required)} requirement(s) not met",
                    missing_requirement_ids=missing,
                    requirements=requirements,
                    workspace_warning=ws_warning or None,
                )

    # Cache for incremental evaluation - copy input to detect mutations
    from copy import deepcopy
    _last_input = deepcopy(inp)
    _last_result = result
    return result
