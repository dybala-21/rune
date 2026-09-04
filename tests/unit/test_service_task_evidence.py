"""R09-R11 judged a service lifecycle that nothing ever reported.

`_execute_managed_service` runs spawn -> readiness -> smoke -> teardown and
ships the outcome of each phase in `CapabilityResult.metadata`, but the loop
never collected it, so `service_task` stayed None and the three service
requirements printed "done" in the trace having checked nothing.

Wiring them is safe in a way it would not be for an arbitrary lifecycle:
readiness must succeed for the call to succeed at all, and teardown runs in a
`finally`, so a successful managed_service run has by construction probed and
cleaned up. The requirements only activate when a service actually ran.
"""

from __future__ import annotations

from rune.agent.completion_gate import (
    CompletionGateInput,
    ExecutionEvidenceSnapshot,
    ServiceTaskEvidenceSnapshot,
    evaluate_completion_gate,
)
from rune.agent.loop import ServiceEvidence

SERVICE_IDS = {"R09_SERVICE_START", "R10_SERVICE_PROBE", "R11_SERVICE_CLEANUP"}


def _lifecycle(readiness=True, smoke=None, teardown=True):
    meta = {
        "mode": "managed_service",
        "pid": 4242,
        "service_id": "abc123",
        "readiness": {"attempted": True, "success": readiness},
        "teardown": {"attempted": True, "success": teardown},
    }
    meta["smoke"] = (
        {"attempted": False, "success": False}
        if smoke is None
        else {"attempted": True, "success": smoke}
    )
    return meta


# ---------------------------------------------------------------------------
# Collecting the evidence
# ---------------------------------------------------------------------------

def test_a_healthy_lifecycle_yields_all_three_signals():
    ev = ServiceEvidence()

    ev.observe(_lifecycle())

    assert ev.starts == 1
    assert ev.runtime_probes >= 1
    assert ev.cleanups == 1


def test_a_smoke_check_counts_as_a_second_probe():
    ev = ServiceEvidence()

    ev.observe(_lifecycle(smoke=True))

    assert ev.runtime_probes == 2


def test_a_service_that_never_became_ready_records_no_probe():
    ev = ServiceEvidence()

    ev.observe(_lifecycle(readiness=False, teardown=True))

    assert ev.starts == 1
    assert ev.runtime_probes == 0


def test_incomplete_cleanup_is_not_recorded_as_a_cleanup():
    ev = ServiceEvidence()

    ev.observe(_lifecycle(teardown=False))

    assert ev.cleanups == 0


def test_ordinary_bash_is_not_a_service():
    ev = ServiceEvidence()

    ev.observe({"exit_code": 0, "cwd": "/project"})
    ev.observe(None)
    ev.observe({})

    assert ev.observed is False
    assert ev.starts == 0


def test_repeated_services_accumulate():
    ev = ServiceEvidence()

    ev.observe(_lifecycle())
    ev.observe(_lifecycle(smoke=True))

    assert ev.starts == 2
    assert ev.cleanups == 2


# ---------------------------------------------------------------------------
# What the gate then does with it
# ---------------------------------------------------------------------------

def _gate(service_task):
    return evaluate_completion_gate(
        CompletionGateInput(
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(executions=1),
            answer_length=200,
            service_task=service_task,
        )
    )


def test_without_a_service_the_requirements_stay_inactive():
    """A task that started no service must not be judged as a service task."""
    result = _gate(None)

    for req in result.requirements:
        if req.id in SERVICE_IDS:
            assert req.required is False


def test_a_healthy_service_satisfies_all_three():
    ev = ServiceEvidence()
    ev.observe(_lifecycle(smoke=True))

    result = _gate(ServiceTaskEvidenceSnapshot(**ev.as_snapshot_kwargs()))

    for req in result.requirements:
        if req.id in SERVICE_IDS:
            assert req.required is True, req.id
            assert req.status == "done", f"{req.id}: {req.failure_reason}"


def test_a_service_that_never_became_ready_is_reported_not_verified():
    ev = ServiceEvidence()
    ev.observe(_lifecycle(readiness=False))

    result = _gate(ServiceTaskEvidenceSnapshot(**ev.as_snapshot_kwargs()))

    assert "R10_SERVICE_PROBE" in result.missing_requirement_ids
    assert result.outcome != "verified"


def test_incomplete_cleanup_is_surfaced_without_blocking():
    """A leaked process is worth reporting, but the work itself may be done.

    R11 is deliberately non-blocking: it reports "skipped" with a reason rather
    than failing the run, and stays out of missing_requirement_ids.
    """
    ev = ServiceEvidence()
    ev.observe(_lifecycle(teardown=False))

    result = _gate(ServiceTaskEvidenceSnapshot(**ev.as_snapshot_kwargs()))
    r11 = next(r for r in result.requirements if r.id == "R11_SERVICE_CLEANUP")

    assert r11.required is True
    assert r11.status == "skipped"
    assert r11.failure_reason
    assert "R11_SERVICE_CLEANUP" not in result.missing_requirement_ids


# ---------------------------------------------------------------------------
# R12: where commands actually ran.
#
# The workspace snapshot only clones the workspace directory, so a command run
# outside it cannot be rolled back. R12 was meant to notice, but the loop never
# passed a workspace snapshot and the warning it produces was read by nobody.
#
# It is wired as a warning, not a block: running outside the workspace is
# routine and legitimate (temp dirs, git worktrees), so blocking on it would
# refuse correct work. Going from "never evaluated" to "reported" is strictly
# stricter than before.
# ---------------------------------------------------------------------------

from rune.agent.completion_gate import WorkspaceAlignmentSnapshot
from rune.agent.loop import ExecutionRoots


def test_commands_without_an_explicit_cwd_are_not_roots():
    """No cwd means the workspace default; there is nothing to misalign."""
    roots = ExecutionRoots()

    roots.observe({"command": "ls -la"})
    roots.observe(None)

    assert roots.roots == []


def test_an_explicit_cwd_is_recorded_once():
    roots = ExecutionRoots()

    roots.observe({"command": "ls", "cwd": "/project/src"})
    roots.observe({"command": "pwd", "cwd": "/project/src"})

    assert roots.roots == ["/project/src"]


def test_work_inside_the_workspace_raises_no_warning():
    roots = ExecutionRoots()
    roots.observe({"command": "pytest", "cwd": "/project/tests"})

    result = evaluate_completion_gate(
        CompletionGateInput(
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(executions=1),
            answer_length=200,
            workspace=WorkspaceAlignmentSnapshot(
                workspace_root="/project", execution_roots=roots.roots
            ),
        )
    )

    assert result.workspace_warning is None
    r12 = next(r for r in result.requirements if r.id == "R12_WORKSPACE_ALIGNMENT")
    assert r12.required is True
    assert r12.status == "done"


def test_work_outside_the_workspace_warns_without_blocking():
    roots = ExecutionRoots()
    roots.observe({"command": "rm -rf x", "cwd": "/tmp/scratch"})

    result = evaluate_completion_gate(
        CompletionGateInput(
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(executions=1),
            answer_length=200,
            workspace=WorkspaceAlignmentSnapshot(
                workspace_root="/project", execution_roots=roots.roots
            ),
        )
    )
    r12 = next(r for r in result.requirements if r.id == "R12_WORKSPACE_ALIGNMENT")

    assert r12.status == "skipped"
    assert "/tmp/scratch" in (result.workspace_warning or "")
    assert "R12_WORKSPACE_ALIGNMENT" not in result.missing_requirement_ids
    assert result.outcome == "verified", "a temp dir must not fail an otherwise good run"


# ---------------------------------------------------------------------------
# The warning has to reach a person. It was produced by the gate and read by
# nobody — one more channel that ran and changed nothing.
# ---------------------------------------------------------------------------

def test_the_trace_can_carry_the_warning():
    from rune.types import CompletionTrace

    trace = CompletionTrace()

    assert trace.workspace_warning == ""
    trace.workspace_warning = "Execution roots not under workspace: /tmp/x"
    assert trace.workspace_warning


def test_the_trust_payload_carries_it_to_the_app():
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    trace = CompletionTrace(reason="completed")
    trace.workspace_warning = "Execution roots not under workspace: /tmp/x"

    payload = build_trust_payload(trace)

    assert payload["workspaceWarning"] == trace.workspace_warning
    assert payload["verified"] is True, "a temp dir must not flip the verdict"


def test_a_clean_run_sends_no_warning():
    from rune.api.server import build_trust_payload
    from rune.types import CompletionTrace

    payload = build_trust_payload(CompletionTrace(reason="completed"))

    assert payload["workspaceWarning"] == ""


# ---------------------------------------------------------------------------
# Non-blocking statuses must not be handed to the agent as work to finish.
# R11 and R12 report "skipped"; before this they were pulled into the "not met"
# list purely because they were not "done".
# ---------------------------------------------------------------------------

def test_skipped_requirements_are_not_reported_as_missing():
    ev = ServiceEvidence()
    ev.observe(_lifecycle(teardown=False))

    result = evaluate_completion_gate(
        CompletionGateInput(
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(executions=1),
            answer_length=200,
            service_task=ServiceTaskEvidenceSnapshot(**ev.as_snapshot_kwargs()),
            workspace=WorkspaceAlignmentSnapshot(
                workspace_root="/p", execution_roots=["/p/src", "/tmp/x"]
            ),
        )
    )

    skipped = {r.id for r in result.requirements if r.status == "skipped"}
    assert skipped, "expected the non-blocking statuses to be exercised"
    assert not (skipped & set(result.missing_requirement_ids)), (
        "a status the gate does not block on was reported as missing"
    )
    assert result.outcome == "verified"
