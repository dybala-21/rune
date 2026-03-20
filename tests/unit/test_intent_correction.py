"""Tests for rune.agent.intent_correction — post-execution intent correction."""


from rune.agent.intent_correction import (
    IntentCorrectionServiceEvidence,
    IntentCorrectionSignals,
    apply_post_execution_intent_corrections,
)
from rune.agent.intent_engine import IntentContract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code_write_intent() -> IntentContract:
    return IntentContract(
        kind="code_write",
        tool_requirement="write",
        grounding_requirement="none",
        output_expectation="file",
        requires_code_verification=True,
    )


def make_signals(**kwargs) -> IntentCorrectionSignals:
    defaults = dict(
        structured_write_count=0,
        changed_files_count=0,
        write_evidence=0,
        browser_write_evidence=0,
        verification_evidence=0,
        service_task_evidence=IntentCorrectionServiceEvidence(),
        web_search_evidence=0,
        text_output_length=0,
    )
    defaults.update(kwargs)
    return IntentCorrectionSignals(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntentCorrection:
    def test_downgrades_code_write_to_mixed_no_structured_writes(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(changed_files_count=1, write_evidence=1),
        )
        assert "code_write_no_structured" in result.corrections
        assert result.intent.kind == "mixed"
        assert result.intent.requires_code_verification is False

    def test_downgrades_file_output_for_service_task(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(
                verification_evidence=1,
                service_task_evidence=IntentCorrectionServiceEvidence(
                    starts=1, runtime_probes=1, cleanups=1,
                ),
            ),
        )
        assert "execution_only_service_task" in result.corrections
        assert result.intent.kind == "mixed"
        assert result.intent.output_expectation == "text"
        assert result.intent.requires_code_verification is False

    def test_keeps_file_output_when_no_service_evidence(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(verification_evidence=1),
        )
        assert "execution_only_service_task" not in result.corrections
        assert result.intent.output_expectation == "file"

    def test_keeps_file_output_when_file_artifacts_exist(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(
                structured_write_count=1,
                changed_files_count=1,
                write_evidence=1,
                verification_evidence=1,
                service_task_evidence=IntentCorrectionServiceEvidence(
                    starts=1, runtime_probes=1, cleanups=1,
                ),
            ),
        )
        assert "execution_only_service_task" not in result.corrections
        assert result.intent.output_expectation == "file"

    def test_downgrades_to_text_for_research_with_web_search(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(web_search_evidence=3, text_output_length=2000),
        )
        assert "research_text_output" in result.corrections
        assert result.intent.output_expectation == "text"
        assert result.intent.tool_requirement == "read"

    def test_keeps_file_when_web_search_but_short_text(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(web_search_evidence=2, text_output_length=100),
        )
        assert "research_text_output" not in result.corrections
        assert result.intent.output_expectation == "file"

    def test_keeps_file_when_artifacts_exist_despite_web_search(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(
                structured_write_count=1,
                changed_files_count=1,
                write_evidence=1,
                web_search_evidence=5,
                text_output_length=3000,
            ),
        )
        assert "research_text_output" not in result.corrections
        assert result.intent.output_expectation == "file"

    def test_preserves_non_write_tool_requirement(self):
        intent = IntentContract(
            kind="mixed",
            tool_requirement="read",
            grounding_requirement="none",
            output_expectation="file",
            requires_code_verification=False,
        )
        result = apply_post_execution_intent_corrections(
            intent,
            make_signals(web_search_evidence=2, text_output_length=1000),
        )
        assert "research_text_output" in result.corrections
        assert result.intent.tool_requirement == "read"

    def test_no_corrections_when_all_conditions_normal(self):
        result = apply_post_execution_intent_corrections(
            code_write_intent(),
            make_signals(
                structured_write_count=2,
                changed_files_count=3,
                write_evidence=2,
            ),
        )
        assert len(result.corrections) == 0
        assert result.intent.kind == "code_write"
