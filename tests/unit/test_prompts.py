"""Tests for the prompts module."""

from __future__ import annotations

from unittest.mock import MagicMock

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    PROMPT_BROWSER,
    PROMPT_CODE,
    PROMPT_CORE,
    PROMPT_DOCUMENT,
    PROMPT_EMAIL_WORKFLOW,
    PROMPT_SERVICE_SAFETY,
    PROMPT_WEB_BASE,
    PROMPT_WEB_DEEP,
    PROMPT_WEB_EFFICIENCY,
    build_continuation_prompt,
    build_system_prompt,
)


class TestPromptConstants:
    """Verify modular prompt sections contain expected content."""

    def test_core_contains_language_rule(self):
        assert "LANGUAGE RULE" in PROMPT_CORE

    def test_core_contains_principles(self):
        assert "Core Principles" in PROMPT_CORE

    def test_core_contains_task_completion(self):
        assert "Task Completion" in PROMPT_CORE

    def test_core_contains_error_recovery(self):
        assert "Error Recovery" in PROMPT_CORE

    def test_code_contains_delegation(self):
        assert "Delegation" in PROMPT_CODE

    def test_code_contains_analysis_strategy(self):
        assert "Project Analysis Strategy" in PROMPT_CODE

    def test_web_base_contains_escalation(self):
        assert "Escalation path" in PROMPT_WEB_BASE

    def test_web_base_contains_hallucination_prevention(self):
        assert "NEVER hallucinate" in PROMPT_WEB_BASE

    def test_web_efficiency_contains_step_target(self):
        assert "2-4 steps" in PROMPT_WEB_EFFICIENCY

    def test_web_deep_contains_research_protocol(self):
        assert "Adaptive Research Depth" in PROMPT_WEB_DEEP

    def test_browser_contains_profiles(self):
        assert "Browser Tools" in PROMPT_BROWSER

    def test_browser_contains_multi_turn(self):
        assert "Multi-Turn" in PROMPT_BROWSER

    def test_document_contains_self_review(self):
        assert "Self-Review" in PROMPT_DOCUMENT

    def test_email_workflow_contains_gmail(self):
        assert "Gmail" in PROMPT_EMAIL_WORKFLOW

    def test_service_safety_contains_oauth(self):
        assert "OAuth" in PROMPT_SERVICE_SAFETY

    def test_legacy_alias(self):
        """AGENT_SYSTEM_PROMPT is an alias for PROMPT_CORE."""
        assert AGENT_SYSTEM_PROMPT is PROMPT_CORE


class TestBuildSystemPrompt:
    """Test conditional assembly logic."""

    def test_always_includes_core(self):
        prompt = build_system_prompt(goal="test")
        assert "LANGUAGE RULE" in prompt
        assert "Core Principles" in prompt

    def test_includes_goal(self):
        goal_text = "Refactor the authentication module"
        prompt = build_system_prompt(goal=goal_text)
        assert goal_text in prompt
        assert "Current Task" in prompt

    def test_code_category_includes_prompt_code(self):
        prompt = build_system_prompt(goal="fix bug", goal_category="code")
        assert "Code Intelligence" in prompt
        assert "Delegation" in prompt

    def test_code_category_excludes_browser(self):
        prompt = build_system_prompt(goal="fix bug", goal_category="code")
        assert "Browser Tools" not in prompt

    def test_web_category_includes_web_prompts(self):
        prompt = build_system_prompt(goal="search news", goal_category="web")
        assert "Escalation path" in prompt
        assert "Step Efficiency" in prompt

    def test_web_category_excludes_code(self):
        prompt = build_system_prompt(goal="search news", goal_category="web")
        assert "Code Intelligence" not in prompt

    def test_full_category_includes_everything(self):
        prompt = build_system_prompt(goal="big task", goal_category="full")
        assert "Code Intelligence" in prompt
        assert "Escalation path" in prompt
        assert "Browser Tools" in prompt

    def test_deep_research_includes_deep_excludes_efficiency(self):
        prompt = build_system_prompt(
            goal="research topic", goal_category="web", is_deep_research=True
        )
        assert "Research Protocol" in prompt
        assert "Escalation path" in prompt
        # Efficiency prompt should NOT be included in deep research
        assert "Step Efficiency" not in prompt

    def test_browser_deferred_excludes_browser(self):
        prompt = build_system_prompt(
            goal="browse site", goal_category="full", defer_browser=True
        )
        assert "Browser Tools" not in prompt

    def test_browser_not_deferred_includes_browser(self):
        prompt = build_system_prompt(
            goal="browse site", goal_category="browser", defer_browser=False
        )
        assert "Browser Tools" in prompt

    def test_mcp_services_includes_service_safety(self):
        prompt = build_system_prompt(goal="sync calendar", has_mcp_services=True)
        assert "External Service Operations" in prompt

    def test_no_mcp_excludes_service_safety(self):
        prompt = build_system_prompt(goal="sync calendar", has_mcp_services=False)
        assert "External Service Operations" not in prompt

    def test_email_goal_includes_email_workflow(self):
        prompt = build_system_prompt(goal="check my gmail inbox")
        assert "Email Reading Workflow" in prompt

    def test_non_email_goal_excludes_email_workflow(self):
        prompt = build_system_prompt(goal="fix the CSS layout")
        assert "Email Reading Workflow" not in prompt

    def test_document_goal_includes_document_protocol(self):
        prompt = build_system_prompt(goal="사업 계획서 작성해줘")
        assert "Document Creation Protocol" in prompt

    def test_non_document_goal_excludes_document_protocol(self):
        prompt = build_system_prompt(goal="fix the bug")
        assert "Document Creation Protocol" not in prompt

    def test_channel_telegram_includes_channel_rules(self):
        prompt = build_system_prompt(goal="test", channel="telegram")
        assert "Channel: telegram" in prompt
        assert "ALWAYS use tools" in prompt

    def test_channel_tui_excludes_channel_rules(self):
        prompt = build_system_prompt(goal="test", channel="tui")
        assert "Channel:" not in prompt

    def test_channel_cli_excludes_channel_rules(self):
        prompt = build_system_prompt(goal="test", channel="cli")
        assert "Channel:" not in prompt

    def test_channel_none_excludes_channel_rules(self):
        prompt = build_system_prompt(goal="test", channel=None)
        assert "Channel:" not in prompt

    def test_environment_injection(self):
        prompt = build_system_prompt(
            goal="test",
            environment={"cwd": "/home/user/project", "home": "/home/user"},
        )
        assert "Environment" in prompt
        assert "/home/user/project" in prompt
        assert "Home directory: /home/user" in prompt

    def test_environment_includes_datetime(self):
        prompt = build_system_prompt(
            goal="test",
            environment={"cwd": "/tmp"},
        )
        assert "Current date and time:" in prompt

    def test_repo_map_injection(self):
        prompt = build_system_prompt(goal="analyze", repo_map="src/\n  main.py (50)")
        assert "Repository Map" in prompt
        assert "src/" in prompt

    def test_classification_with_goal_type(self):
        cls = ClassificationResult(
            goal_type="code_modify", confidence=0.9, tier=1
        )
        prompt = build_system_prompt(goal="fix bug", classification=cls)
        assert "code_modify" in prompt
        assert "Task Classification" in prompt

    def test_memory_context(self):
        memory = MagicMock()
        memory.formatted = "Previously worked on auth module"
        prompt = build_system_prompt(goal="continue", memory_context=memory)
        assert "Memory Context" in prompt
        assert "Previously worked on auth module" in prompt

    def test_knowledge_inventory(self):
        prompt = build_system_prompt(
            goal="analyze",
            knowledge_inventory="Files read: 5\nSearches: 3",
        )
        assert "Knowledge Inventory" in prompt
        assert "Files read: 5" in prompt

    def test_no_optional_sections_when_none(self):
        prompt = build_system_prompt(goal="simple task", goal_category="code")
        assert "Task Classification" not in prompt
        assert "Memory Context" not in prompt
        assert "Knowledge Inventory" not in prompt

    def test_file_output_expectation(self):
        cls = ClassificationResult(
            goal_type="full",
            confidence=0.9,
            tier=2,
            output_expectation="file",
        )
        prompt = build_system_prompt(
            goal="write report",
            classification=cls,
            environment={"cwd": "/tmp/project"},
        )
        assert "File Output Required" in prompt
        assert "/tmp/project" in prompt

    def test_continuation_prompt_injection(self):
        cls = ClassificationResult(
            goal_type="code_modify",
            confidence=0.9,
            tier=2,
            is_continuation=True,
        )
        prompt = build_system_prompt(goal="계속해", classification=cls)
        assert "Follow-up Task Scope Control" in prompt

    def test_multi_phase_prompt_injection(self):
        cls = ClassificationResult(
            goal_type="code_modify",
            confidence=0.9,
            tier=2,
            is_continuation=True,
            is_complex_coding=True,
        )
        prompt = build_system_prompt(goal="계속해", classification=cls)
        assert "Multi-Phase Continuation Protocol" in prompt

    def test_multi_task_prompt_injection(self):
        cls = ClassificationResult(
            goal_type="full",
            confidence=0.9,
            tier=2,
            is_multi_task=True,
        )
        prompt = build_system_prompt(goal="A하고 B하고 C해줘", classification=cls)
        assert "Multi-Task Execution Protocol" in prompt

    def test_complex_task_prompt_injection(self):
        cls = ClassificationResult(
            goal_type="full",
            confidence=0.9,
            tier=2,
            is_complex_coding=True,
        )
        prompt = build_system_prompt(goal="build full app", classification=cls)
        assert "Complex Task Execution Protocol" in prompt

    def test_execution_mode_injection(self):
        cls = ClassificationResult(
            goal_type="execution",
            confidence=0.9,
            tier=2,
            requires_execution=True,
        )
        prompt = build_system_prompt(goal="fix and run", classification=cls)
        assert "Execution Mode (MANDATORY)" in prompt

    def test_complex_coding_not_duplicated_with_multi_task(self):
        """Complex coding protocol should NOT appear when multi_task is also set."""
        cls = ClassificationResult(
            goal_type="full",
            confidence=0.9,
            tier=2,
            is_complex_coding=True,
            is_multi_task=True,
        )
        prompt = build_system_prompt(goal="big task", classification=cls)
        assert "Multi-Task Execution Protocol" in prompt
        assert "Complex Task Execution Protocol" not in prompt


class TestBuildContinuationPrompt:
    def test_basic_continuation(self):
        result = build_continuation_prompt(
            reason="Verification not complete",
            evidence="2 of 5 tests passing",
        )
        assert isinstance(result, str)
        assert "Continuation Required" in result
        assert "Verification not complete" in result
        assert "2 of 5 tests passing" in result
        assert "Continue working" in result

    def test_continuation_without_evidence(self):
        result = build_continuation_prompt(reason="Task incomplete")
        assert "Task incomplete" in result
        assert "Continue working" in result
        # Evidence section should not appear
        assert "Evidence so far" not in result

    def test_contains_no_repeat_instruction(self):
        result = build_continuation_prompt(reason="more work needed")
        assert "Do NOT repeat" in result


class TestClassificationResultExtendedFields:
    """Verify new fields on ClassificationResult."""

    def test_default_values(self):
        cls = ClassificationResult(goal_type="chat", confidence=0.8, tier=1)
        assert cls.is_continuation is False
        assert cls.is_complex_coding is False
        assert cls.is_multi_task is False
        assert cls.requires_code is False
        assert cls.requires_execution is False
        assert cls.complexity == "simple"
        assert cls.output_expectation == "text"

    def test_custom_values(self):
        cls = ClassificationResult(
            goal_type="full",
            confidence=0.95,
            tier=2,
            is_continuation=True,
            is_complex_coding=True,
            is_multi_task=False,
            requires_code=True,
            requires_execution=True,
            complexity="complex",
            output_expectation="file",
        )
        assert cls.is_continuation is True
        assert cls.is_complex_coding is True
        assert cls.requires_code is True
        assert cls.complexity == "complex"
        assert cls.output_expectation == "file"
