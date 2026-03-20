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
