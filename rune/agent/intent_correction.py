"""Intent correction - post-execution intent contract adjustment.

Ported from src/agent/intent-correction.ts (96 lines) - corrects the
intent contract based on observed execution evidence to reduce false
blocks from the completion gate.

Corrections:
- code_write with no structured writes -> mixed
- Execution-only service tasks -> text output expectation
- Research with text output but no file artifacts -> text expectation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rune.agent.intent_engine import IntentContract
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

CorrectionType = Literal[
    "code_write_no_structured",
    "execution_only_service_task",
    "research_text_output",
]


@dataclass(slots=True)
class IntentCorrectionServiceEvidence:
    """Evidence for service-lifecycle tasks."""

    starts: int = 0
    runtime_probes: int = 0
    cleanups: int = 0


@dataclass(slots=True)
class IntentCorrectionSignals:
    """Observation signals used for post-execution intent correction."""

    structured_write_count: int = 0
    changed_files_count: int = 0
    write_evidence: int = 0
    browser_write_evidence: int = 0
    verification_evidence: int = 0
    service_task_evidence: IntentCorrectionServiceEvidence = field(
        default_factory=IntentCorrectionServiceEvidence
    )
    web_search_evidence: int = 0
    text_output_length: int = 0


@dataclass(slots=True)
class IntentCorrectionResult:
    """Result of post-execution intent correction."""

    intent: IntentContract
    corrections: list[CorrectionType] = field(default_factory=list)


# apply_post_execution_intent_corrections

def apply_post_execution_intent_corrections(
    intent: IntentContract,
    signals: IntentCorrectionSignals,
) -> IntentCorrectionResult:
    """Correct intent contract based on observed execution evidence.

    Goal: distinguish code modification from execution/verification tasks
    to reduce completion gate false positives.
    """
    corrected = intent
    corrections: list[CorrectionType] = []

    # Correction 1: code_write with no structured writes but file changes
    if (
        corrected.kind == "code_write"
        and signals.structured_write_count == 0
        and signals.changed_files_count > 0
    ):
        corrected = IntentContract(
            kind="mixed",
            tool_requirement=corrected.tool_requirement,
            grounding_requirement=corrected.grounding_requirement,
            output_expectation=corrected.output_expectation,
            requires_code_verification=False,
        )
        corrections.append("code_write_no_structured")

    # Check for service lifecycle evidence
    svc = signals.service_task_evidence
    has_service_lifecycle = svc.starts > 0 or svc.runtime_probes > 0 or svc.cleanups > 0
    has_file_artifact = (
        signals.changed_files_count > 0
        or signals.structured_write_count > 0
        or signals.write_evidence > 0
        or signals.browser_write_evidence > 0
    )

    # Correction 2: execution-only service task (server start + probe + test)
    # Forcing output=file would cause false block
    if (
        corrected.output_expectation == "file"
        and not has_file_artifact
        and has_service_lifecycle
        and signals.verification_evidence > 0
    ):
        corrected = IntentContract(
            kind="mixed" if corrected.kind == "code_write" else corrected.kind,
            tool_requirement=corrected.tool_requirement,
            grounding_requirement=corrected.grounding_requirement,
            output_expectation="text",
            requires_code_verification=False,
        )
        corrections.append("execution_only_service_task")

    # Correction 3: research/doc work with web searches + text output but no files
    is_research_text_output = (
        corrected.output_expectation == "file"
        and not has_file_artifact
        and not has_service_lifecycle
        and signals.web_search_evidence > 0
        and signals.text_output_length > 500
    )

    if is_research_text_output:
        corrected = IntentContract(
            kind=corrected.kind,
            tool_requirement=(
                "read" if corrected.tool_requirement == "write" else corrected.tool_requirement
            ),
            grounding_requirement=corrected.grounding_requirement,
            output_expectation="text",
            requires_code_verification=corrected.requires_code_verification,
        )
        corrections.append("research_text_output")

    return IntentCorrectionResult(intent=corrected, corrections=corrections)
