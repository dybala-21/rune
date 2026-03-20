"""Tests for rune.agent.completion_trace_format — trace formatting."""


from rune.agent.completion_trace_format import (
    CompletionContractTrace,
    CompletionEvidenceTrace,
    CompletionPlanTrace,
    CompletionRequirementTrace,
    CompletionTraceDisplay,
    FinalAgentResponseInput,
    format_completion_trace_for_user,
    format_final_agent_response_for_user,
    summarize_completion_trace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trace(**kwargs) -> CompletionTraceDisplay:
    defaults = dict(
        outcome="verified",
        requirements=[
            CompletionRequirementTrace(
                id="intent_write_evidence",
                description="Write evidence required",
                required=True,
                status="done",
            ),
            CompletionRequirementTrace(
                id="verification_after_changes",
                description="Verification required",
                required=True,
                status="done",
            ),
            CompletionRequirementTrace(
                id="hard_failure_signals",
                description="Hard failure signals",
                required=True,
                status="done",
            ),
        ],
    )
    defaults.update(kwargs)
    return CompletionTraceDisplay(**defaults)


# ---------------------------------------------------------------------------
# summarize_completion_trace
# ---------------------------------------------------------------------------

class TestSummarizeCompletionTrace:
    def test_returns_none_for_none_trace(self):
        assert summarize_completion_trace(None) is None

    def test_returns_none_when_only_hard_failure_signals(self):
        trace = CompletionTraceDisplay(
            outcome="verified",
            requirements=[
                CompletionRequirementTrace(
                    id="hard_failure_signals",
                    description="Hard failure signals",
                    required=True,
                    status="done",
                ),
            ],
        )
        assert summarize_completion_trace(trace) is None

    def test_counts_done_requirements(self):
        trace = make_trace(
            outcome="partial",
            requirements=[
                CompletionRequirementTrace(
                    id="write_ev", description="Write", required=True, status="done",
                ),
                CompletionRequirementTrace(
                    id="verify", description="Verify", required=True, status="blocked",
                    failure_reason="verification not run",
                ),
                CompletionRequirementTrace(
                    id="hard_failure_signals", description="HF", required=True, status="done",
                ),
            ],
        )
        summary = summarize_completion_trace(trace)
        assert summary is not None
        assert summary.required_done == 1
        assert summary.required_total == 2

    def test_returns_none_for_empty_requirements(self):
        trace = CompletionTraceDisplay(outcome="verified", requirements=[])
        assert summarize_completion_trace(trace) is None


# ---------------------------------------------------------------------------
# format_completion_trace_for_user
# ---------------------------------------------------------------------------

class TestFormatCompletionTraceForUser:
    def test_returns_none_for_none_trace(self):
        assert format_completion_trace_for_user(None) is None

    def test_includes_blocked_details(self):
        trace = make_trace(
            outcome="partial",
            requirements=[
                CompletionRequirementTrace(
                    id="write_ev", description="Write", required=True, status="done",
                ),
                CompletionRequirementTrace(
                    id="verify", description="Verify", required=True, status="blocked",
                    failure_reason="verification not run",
                ),
            ],
        )
        text = format_completion_trace_for_user(trace, include_blocked_details=True)
        assert text is not None
        assert "verification not run" in text

    def test_includes_contract_line(self):
        trace = make_trace(
            outcome="partial",
            contract=CompletionContractTrace(
                kind="code_write",
                tool_requirement="write",
                grounding_requirement="none",
                source="tier2",
                resolved=True,
            ),
        )
        text = format_completion_trace_for_user(trace)
        assert text is not None
        assert "code_write" in text
        assert "tool=write" in text
        assert "resolved:tier2" in text

    def test_includes_plan_line(self):
        trace = make_trace(
            outcome="partial",
            contract_plan=CompletionPlanTrace(
                action_plan=["step1", "step2"],
                completion_criteria=["criterion1"],
                verification_candidates=["npm test", "tsc"],
            ),
        )
        text = format_completion_trace_for_user(trace)
        assert text is not None
        assert "2 steps" in text or "Plan:" in text

    def test_includes_workspace_line_when_not_verified(self):
        trace = make_trace(
            outcome="partial",
            workspace_root="/repo/demo",
        )
        text = format_completion_trace_for_user(trace)
        assert text is not None
        assert "/repo/demo" in text

    def test_includes_workspace_warning(self):
        trace = make_trace(
            outcome="partial",
            workspace_warning="workspace mismatch detected",
        )
        text = format_completion_trace_for_user(trace)
        assert text is not None
        assert "workspace mismatch" in text

    def test_includes_evidence_line_when_outstanding(self):
        trace = make_trace(
            outcome="partial",
            requirements=[
                CompletionRequirementTrace(
                    id="write_ev", description="Write", required=True, status="blocked",
                ),
            ],
            evidence=CompletionEvidenceTrace(reads=3, writes=1, executions=2),
        )
        text = format_completion_trace_for_user(trace)
        assert text is not None
        assert "read=" in text or "write=" in text


# ---------------------------------------------------------------------------
# format_final_agent_response_for_user
# ---------------------------------------------------------------------------

class TestFormatFinalAgentResponseForUser:
    def test_returns_answer_on_success(self):
        result = format_final_agent_response_for_user(
            FinalAgentResponseInput(success=True, answer="All done!"),
        )
        assert "All done!" in result

    def test_returns_error_on_failure(self):
        result = format_final_agent_response_for_user(
            FinalAgentResponseInput(success=False, error="Something failed"),
        )
        assert "Something failed" in result

    def test_default_message_when_empty(self):
        result = format_final_agent_response_for_user(
            FinalAgentResponseInput(success=True),
        )
        assert "Task completed" in result

    def test_includes_compaction_notice(self):
        result = format_final_agent_response_for_user(
            FinalAgentResponseInput(
                success=True,
                answer="Done",
                show_compaction_notice=True,
                compaction_notices=["context truncated"],
            ),
        )
        assert "context truncated" in result
        assert "Done" in result
