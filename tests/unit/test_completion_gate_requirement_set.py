"""The requirement set must not carry entries that report "done" unchecked.

Every requirement is ``(not required) or <evidence check>``. One whose
``required`` can never become true is not lenient — it is a line in the
user-visible trace claiming a check happened when none did.

R16 (module coverage) and R17 (deep analysis) were removed on that basis:
R16 gated on ev.unique_file_reads, the same signal R15 already uses, with
module_count never entering the ratio its label promised; R17 had no
definition of a "deep analysis tool" outside the gate. The research says
neither is worth rebuilding — see the note in completion_gate.py.
"""

from __future__ import annotations

import pytest

from rune.agent.completion_gate import (
    CompletionGateInput,
    ExecutionEvidenceSnapshot,
    evaluate_completion_gate,
)

REMOVED_IDS = {"R16_MODULE_COVERAGE", "R17_DEEP_ANALYSIS"}


def _trace(**kwargs):
    return evaluate_completion_gate(CompletionGateInput(**kwargs)).requirements


def test_removed_requirements_are_gone_from_the_trace():
    ids = {r.id for r in _trace(
        intent_resolved=True,
        evidence=ExecutionEvidenceSnapshot(reads=3),
        answer_length=200,
    )}

    assert not (ids & REMOVED_IDS), f"still present: {ids & REMOVED_IDS}"


def test_the_removed_inputs_no_longer_exist():
    """A caller passing them should fail loudly rather than be silently ignored."""
    for field in (
        "module_count",
        "min_module_coverage",
        "deep_analysis_tools",
        "min_deep_analysis_tools",
    ):
        with pytest.raises(TypeError):
            CompletionGateInput(intent_resolved=True, **{field: 1})


def test_analysis_depth_still_gates_on_its_floor():
    """R15 keeps the read check R16 duplicated; removing R16 must not lose it."""
    blocked = _trace(
        intent_resolved=True,
        evidence=ExecutionEvidenceSnapshot(reads=1, unique_file_reads=1),
        analysis_depth_min_reads=3,
        answer_length=200,
    )
    r15 = next(r for r in blocked if r.id == "R15_ANALYSIS_DEPTH")

    assert r15.required is True
    assert r15.status == "blocked"

    passing = _trace(
        intent_resolved=True,
        evidence=ExecutionEvidenceSnapshot(reads=5, unique_file_reads=5),
        analysis_depth_min_reads=3,
        answer_length=200,
    )
    assert next(r for r in passing if r.id == "R15_ANALYSIS_DEPTH").status == "done"


def test_analysis_depth_floor_stays_a_floor():
    """Guard the research conclusion: this is 'did you read anything', not coverage.

    Production sets 1-3 on research goals. A higher figure turns a presence
    check into an unevidenced coverage claim and pushes the agent to pad reads.
    """
    import rune.agent.loop as loop

    source = loop.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()

    assigned = {
        int(line.split("=")[1].strip())
        for line in text.splitlines()
        if line.strip().startswith("analysis_min_reads =")
    }

    assert assigned, "analysis_min_reads is no longer assigned in the loop"
    assert max(assigned) <= 3, f"floor raised to {max(assigned)} — see the citations"
