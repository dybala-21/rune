"""Contract plan - contract-based planning with pre/post conditions.

Ported from src/agent/contract-plan.ts (163 lines) - builds an execution
plan trace from an intent contract, including action steps, completion
criteria, and verification candidates based on workspace markers.

buildContractPlanTrace(): Generate a CompletionContractPlanTrace.
detectWorkspaceFiles(): Detect project marker files (go.mod, package.json, etc.).
buildProbeCandidates(): Generate runtime HTTP probe commands per ecosystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rune.agent.intent_engine import IntentContract
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Types

@dataclass(slots=True)
class BuildContractPlanInput:
    """Input for building a contract plan trace."""

    goal: str
    intent: IntentContract
    intent_resolved: bool
    workspace_root: str
    workspace_files: list[str] | None = None


@dataclass(slots=True)
class CompletionContractPlanTrace:
    """Execution plan trace derived from intent contract."""

    objective: str = ""
    action_plan: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)
    verification_candidates: list[str] = field(default_factory=list)
    probe_candidates: list[str] = field(default_factory=list)


# Helpers

def _clip_text(text: str, max_len: int = 140) -> str:
    """Clip text to max length, normalizing whitespace."""
    import re

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


# Workspace detection

WORKSPACE_PROBES = ["go.mod", "package.json", "pyproject.toml", "pytest.ini", "Cargo.toml"]


def detect_workspace_files(workspace_root: str) -> list[str]:
    """Detect project marker files in the workspace root."""
    return [
        f for f in WORKSPACE_PROBES
        if os.path.exists(os.path.join(workspace_root, f))
    ]


# Verification candidates

def _build_verification_candidates(
    workspace_files: list[str],
    requires_code_verification: bool,
) -> list[str]:
    """Build verification command candidates based on workspace markers."""
    if not requires_code_verification:
        return []

    files = set(workspace_files)
    candidates: list[str] = []

    if "go.mod" in files:
        candidates.append("go test ./...")
    if "package.json" in files:
        candidates.append("npm run test")
        candidates.append("npm run typecheck")
    if "pyproject.toml" in files or "pytest.ini" in files:
        candidates.append("pytest")
    if "Cargo.toml" in files:
        candidates.append("cargo test")

    if not candidates:
        candidates.append(
            "Check project Makefile/CI/README for build/test commands and execute"
        )

    return _dedupe(candidates)


# Probe candidates

def build_probe_candidates(
    workspace_files: list[str],
    endpoint: str | None = None,
) -> list[str]:
    """Generate runtime HTTP probe commands per project ecosystem.

    Uses native HTTP clients instead of curl/wget.
    """
    files = set(workspace_files)
    url = endpoint or "http://127.0.0.1:PORT/healthz"
    candidates: list[str] = []

    if "go.mod" in files:
        candidates.append(
            f'go run -e \'package main; import ("fmt";"net/http";"io"); '
            f"func main() {{ r,_:=http.Get(\"{url}\"); "
            f"b,_:=io.ReadAll(r.Body); fmt.Println(string(b)) }}'"
        )
    if "package.json" in files:
        candidates.append(
            f"node -e \"require('http').get('{url}', r => {{ "
            f"let d=''; r.on('data',c=>d+=c); "
            f"r.on('end',()=>console.log(r.statusCode,d)) }})\""
        )
    if "pyproject.toml" in files or "pytest.ini" in files:
        candidates.append(
            f"python3 -c \"import urllib.request; "
            f"print(urllib.request.urlopen('{url}').read().decode())\""
        )
    if "Cargo.toml" in files:
        candidates.append("cargo test (use reqwest or std::net::TcpStream for HTTP probe)")

    return candidates


# Action plan

def _build_action_plan(intent: IntentContract) -> list[str]:
    """Build action plan steps from intent contract."""
    steps: list[str] = []

    if intent.tool_requirement == "none":
        steps.append("Confirm request intent and compose answer")
    elif intent.tool_requirement == "read":
        steps.append("Retrieve relevant context/sources for evidence")
        steps.append("Compose evidence-based result")
    elif intent.tool_requirement == "write":
        if intent.requires_code_verification:
            steps.append("Modify code/config per requirements")
            steps.append("Review change impact scope")
            steps.append("Run verification commands and incorporate results")
        else:
            steps.append("Create/modify files per requirements")
            steps.append("Verify output matches request")

    if intent.grounding_requirement != "none":
        steps.append("Verify against latest/official sources")

    return _dedupe(steps)


# Completion criteria

def _build_completion_criteria(
    intent: IntentContract,
    intent_resolved: bool,
) -> list[str]:
    """Build completion criteria from intent contract."""
    criteria: list[str] = []

    if not intent_resolved:
        criteria.append("Request intent must be in resolved state")

    if intent.tool_requirement == "read":
        criteria.append("At least 1 read tool evidence obtained")
    if intent.tool_requirement == "write":
        criteria.append("At least 1 write/execution evidence obtained")
        if intent.requires_code_verification:
            criteria.append("Run verification command after file changes")

    if intent.grounding_requirement == "required":
        criteria.append("Source verification is required")
    elif intent.grounding_requirement == "recommended":
        criteria.append("Source verification recommended")

    criteria.append("No hard failure signals")

    return _dedupe(criteria)


# Public API

def build_contract_plan_trace(inp: BuildContractPlanInput) -> CompletionContractPlanTrace:
    """Build a complete contract plan trace from input.

    Combines action plan, completion criteria, verification candidates,
    and probe candidates into a single trace object.
    """
    workspace_files = inp.workspace_files or detect_workspace_files(inp.workspace_root)
    probe_candidates = build_probe_candidates(workspace_files)

    return CompletionContractPlanTrace(
        objective=_clip_text(inp.goal),
        action_plan=_build_action_plan(inp.intent),
        completion_criteria=_build_completion_criteria(inp.intent, inp.intent_resolved),
        verification_candidates=_build_verification_candidates(
            workspace_files,
            inp.intent.requires_code_verification,
        ),
        probe_candidates=probe_candidates if probe_candidates else [],
    )
