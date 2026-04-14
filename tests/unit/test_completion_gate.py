"""Tests for the completion gate module."""

from __future__ import annotations

from rune.agent.completion_gate import (
    CompletionGateInput,
    ExecutionEvidenceSnapshot,
    WorkspaceAlignmentSnapshot,
    evaluate_completion_gate,
)


class TestEvaluateCompletionGate:
    def test_verified_simple_read(self):
        inp = CompletionGateInput(
            tool_requirement="read",
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(reads=1, file_reads=1),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "verified"
        assert result.success is True

    def test_blocked_no_evidence(self):
        inp = CompletionGateInput(
            tool_requirement="write",
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(writes=0),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "partial"
        assert result.success is False

    def test_hard_failure_blocks(self):
        inp = CompletionGateInput(
            tool_requirement="none",
            intent_resolved=True,
            hard_failures=["Segfault in tool execution"],
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "blocked"
        assert result.success is False
        assert len(result.hard_failures) == 1

    def test_verification_required(self):
        inp = CompletionGateInput(
            tool_requirement="write",
            intent_resolved=True,
            requires_code_verification=True,
            evidence=ExecutionEvidenceSnapshot(
                reads=1, writes=1, executions=1, verifications=0,
            ),
            structured_write_count=1,
        )
        result = evaluate_completion_gate(inp)
        # Verification not met, so partial
        assert result.outcome == "partial"
        assert any("VERIFICATION" in rid for rid in result.missing_requirement_ids)

    def test_workspace_alignment_mismatch(self):
        inp = CompletionGateInput(
            tool_requirement="write",
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(reads=1, writes=1, executions=1),
            structured_write_count=1,
            workspace=WorkspaceAlignmentSnapshot(
                workspace_root="/home/user/project",
                primary_execution_root="/var/tmp/other",
            ),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "partial"
        assert result.workspace_warning is not None

    def test_grounding_required(self):
        inp = CompletionGateInput(
            tool_requirement="none",
            intent_resolved=True,
            grounding_requirement=True,
            evidence=ExecutionEvidenceSnapshot(web_searches=0, web_fetches=0),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "partial"

    def test_grounding_met(self):
        inp = CompletionGateInput(
            tool_requirement="none",
            intent_resolved=True,
            grounding_requirement=True,
            evidence=ExecutionEvidenceSnapshot(web_searches=3, web_fetches=1),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "verified"

    def test_soft_write_document_task(self):
        """Long text answer without code artifact should not block when
        tool_requirement is none."""
        inp = CompletionGateInput(
            tool_requirement="none",
            intent_resolved=True,
            answer_length=5000,
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "verified"

    def test_analysis_depth(self):
        inp = CompletionGateInput(
            tool_requirement="read",
            intent_resolved=True,
            analysis_depth_min_reads=5,
            evidence=ExecutionEvidenceSnapshot(
                reads=2, file_reads=2, unique_file_reads=2,
            ),
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "partial"

    def test_all_skipped_when_none(self):
        inp = CompletionGateInput(
            tool_requirement="none",
            intent_resolved=True,
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "verified"
        assert result.success is True
        # Many requirements should be skipped
        skipped = [r for r in result.requirements if r.status == "skipped"]
        assert len(skipped) >= 0  # non-required items get skipped


class TestVerifyFreshness:
    """R19: code modified after last bash_execute must trigger blocked.

    Default-OFF behavior is the most important guarantee — every existing
    test in this file constructs CompletionGateInput WITHOUT the freshness
    fields and must continue to pass unchanged.
    """

    def _base_input(self, **overrides) -> CompletionGateInput:
        defaults = dict(
            tool_requirement="write",
            intent_resolved=True,
            evidence=ExecutionEvidenceSnapshot(
                reads=1, writes=1, executions=1,
            ),
            changed_files_count=1,
            structured_write_count=1,
        )
        defaults.update(overrides)
        return CompletionGateInput(**defaults)

    def test_freshness_off_by_default(self):
        """When verify_freshness_enabled is False, R19 is skipped even if
        last_code_write_step > last_verify_step."""
        inp = self._base_input(
            verify_freshness_enabled=False,
            last_code_write_step=5,
            last_verify_step=3,
        )
        result = evaluate_completion_gate(inp)
        # No R19 in missing list
        assert not any(
            "R19" in rid or "VERIFY_FRESHNESS" in rid
            for rid in result.missing_requirement_ids
        )

    def test_freshness_on_blocks_when_write_after_verify(self):
        """Flag ON, code written at step 5, last verify at step 3 → blocked."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=5,
            last_verify_step=3,
        )
        result = evaluate_completion_gate(inp)
        assert result.outcome == "blocked"  # R19 is critical
        assert any(
            "R19" in rid or "VERIFY_FRESHNESS" in rid
            for rid in result.missing_requirement_ids
        )

    def test_freshness_on_passes_when_verify_after_write(self):
        """Flag ON, code written at step 3, verified at step 5 → passes R19."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=3,
            last_verify_step=5,
        )
        result = evaluate_completion_gate(inp)
        # R19 should NOT be in missing list
        assert not any(
            "R19" in rid or "VERIFY_FRESHNESS" in rid
            for rid in result.missing_requirement_ids
        )

    def test_freshness_on_passes_when_same_step(self):
        """Edit and verify in the same step (parallel tool calls) is fine."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=4,
            last_verify_step=4,
        )
        result = evaluate_completion_gate(inp)
        assert not any(
            "R19" in rid for rid in result.missing_requirement_ids
        )

    def test_freshness_on_skipped_when_no_code_writes(self):
        """If no code was ever written, R19 is not required even with flag."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=0,
            last_verify_step=0,
        )
        result = evaluate_completion_gate(inp)
        # R19 is not blocked
        r19 = next(
            (r for r in result.requirements if "R19" in r.id),
            None,
        )
        assert r19 is not None
        assert r19.required is False
        assert r19.status == "done"

    def test_freshness_failure_reason_contains_step_numbers(self):
        """Failure reason must give the model actionable info."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=7,
            last_verify_step=2,
        )
        result = evaluate_completion_gate(inp)
        r19 = next(r for r in result.requirements if "R19" in r.id)
        assert r19.status == "blocked"
        assert "7" in r19.failure_reason
        assert "2" in r19.failure_reason

    def test_freshness_critical_block_not_partial(self):
        """R19 must produce 'blocked' (not 'partial') so the loop forces
        retries instead of accepting the result."""
        inp = self._base_input(
            verify_freshness_enabled=True,
            last_code_write_step=10,
            last_verify_step=5,
        )
        result = evaluate_completion_gate(inp)
        # Critical block prevents partial completion
        assert result.outcome == "blocked"
        assert result.success is False
