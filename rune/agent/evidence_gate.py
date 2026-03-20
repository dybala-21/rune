"""Evidence gate - execution evidence validation.

Ported from src/agent/evidence-gate.ts (36 lines) - thin wrapper around
completion-gate that re-exports execution evidence types and provides
the evaluateEvidenceGate entry point.

Verifies that claimed work actually happened by delegating to the
completion gate's requirement evaluation.
"""

from __future__ import annotations

from rune.agent.completion_gate import (
    CompletionGateInput,
    CompletionGateResult,
    EvidenceSamples,
    ExecutionEvidenceSnapshot,
    evaluate_completion_gate,
)

# Re-export types for downstream consumers
ExecutionEvidence = ExecutionEvidenceSnapshot
EvidenceGateResult = CompletionGateResult

__all__ = [
    "ExecutionEvidence",
    "EvidenceGateResult",
    "EvidenceSamples",
    "CompletionGateInput",
    "evaluate_evidence_gate",
]


def evaluate_evidence_gate(inp: CompletionGateInput) -> EvidenceGateResult:
    """Evaluate the execution evidence gate.

    Delegates to the full completion gate evaluation.
    This was previously inline in loop.py.
    """
    return evaluate_completion_gate(inp)
