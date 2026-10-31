"""Tests for the prompts module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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

    def test_fanout_hint_pinned_to_core(self):
        # The builder removes FANOUT_HINT by exact string match.
        from rune.agent.prompts import FANOUT_HINT
        assert FANOUT_HINT in PROMPT_CORE


class TestFanoutHintSwitch:
    def test_hint_present_by_default(self, monkeypatch):
        from rune.agent.prompts import FANOUT_HINT
        monkeypatch.delenv("RUNE_FANOUT_READS", raising=False)
        assert FANOUT_HINT in build_system_prompt(goal="test")

    def test_hint_stripped_when_env_off(self, monkeypatch):
        from rune.agent.prompts import FANOUT_HINT
        monkeypatch.setenv("RUNE_FANOUT_READS", "0")
        prompt = build_system_prompt(goal="test")
        assert FANOUT_HINT not in prompt
        # Keep the surrounding instruction.
        assert "then read them together in one step." in prompt


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

    def test_memory_context_reaches_prompt_in_all_shapes(self):
        # All supported memory formats must reach the prompt.
        marker = "integer division must floor"
        section = f"## Learned Rules\n- floor: {marker}"

        class _Obj:
            formatted = section

        for ctx in (
            _Obj(),
            {"memory_context": section, "workspace_root": "/x"},
            section,
        ):
            prompt = build_system_prompt(goal="x", memory_context=ctx)
            assert marker in prompt, f"memory dropped for {type(ctx).__name__}"

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

    @pytest.mark.parametrize("deep", [False, True])
    def test_small_lookup_keeps_sources_and_browser_fallback_without_research_overhead(self, deep):
        from rune.agent.prompts import PROMPT_CHAT

        classification = ClassificationResult(goal_type="web", confidence=.99, tier=2,
                                             needs_web_evidence=True, lookup_only=True, complexity="simple")
        prompt = build_system_prompt(goal="Read the current value", classification=classification,
                                     goal_category="web", is_deep_research=deep)
        if deep:
            assert PROMPT_CORE in prompt and PROMPT_WEB_DEEP in prompt
        else:
            assert PROMPT_CHAT in prompt and PROMPT_CORE not in prompt
            assert "Use browser reading tools" in prompt
            assert "observation time" in prompt and "clickable source link" in prompt
            assert "exact-output constraints" in prompt

    def test_web_category_excludes_code(self):
        prompt = build_system_prompt(goal="search news", goal_category="web")
        assert "Code Intelligence" not in prompt

    @pytest.mark.parametrize("deep,lookup", [(False, True), (False, False), (True, False)])
    def test_research_routes_share_selective_reading_and_coverage_guidance(self, deep, lookup):
        classification = ClassificationResult(goal_type="research" if deep else "web", confidence=.99, tier=2,
                                             needs_web_evidence=True, needs_current_source=True, lookup_only=lookup)
        prompt = build_system_prompt("Research recent service notices", classification=classification,
                                     goal_category="web", is_deep_research=deep)
        assert prompt.count("Fetch selectively with a focused selector or a bounded maxLength") == 1
        assert "retrieve the complete content when the user requests it" in prompt
        assert prompt.count("relevant alternatives no longer resolve the remaining gaps") == 1
        assert prompt.count("read the relevant frame URL before opening a browser") == 1
        assert prompt.count("Do not add a calendar year to every discovery query") == 1
        assert "maxLength=8000" not in prompt

    @pytest.mark.parametrize("category,lookup", [("chat", False), ("web", True), ("web", False)])
    def test_source_scope_guidance_survives_prompt_compaction(self, category, lookup):
        classification = ClassificationResult(goal_type=category, confidence=.99, tier=2,
                                             needs_web_evidence=True, needs_current_source=True, lookup_only=lookup)
        prompt = build_system_prompt(goal="Check the current reading", classification=classification,
                                     goal_category=category)
        assert "Do not infer safety or normality from a reading alone" in prompt

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

    def test_email_intent_includes_email_workflow(self):
        classification = ClassificationResult(
            goal_type="full", confidence=0.9, tier=2,
            intent_categories=frozenset({"email"}),
        )
        prompt = build_system_prompt(
            goal="check my gmail inbox", classification=classification,
        )
        assert "Email Reading Workflow" in prompt

    def test_no_email_intent_excludes_email_workflow(self):
        classification = ClassificationResult(
            goal_type="code_modify", confidence=0.9, tier=2,
            intent_categories=frozenset(),
        )
        prompt = build_system_prompt(
            goal="fix the CSS layout", classification=classification,
        )
        assert "Email Reading Workflow" not in prompt

    def test_document_intent_includes_document_protocol(self):
        classification = ClassificationResult(
            goal_type="full", confidence=0.9, tier=2, output_expectation="file",
            intent_categories=frozenset({"document"}),
        )
        prompt = build_system_prompt(
            goal="사업 계획서 작성해줘", classification=classification,
        )
        assert "Document Creation Protocol" in prompt

    def test_no_document_intent_excludes_document_protocol(self):
        classification = ClassificationResult(
            goal_type="code_modify", confidence=0.9, tier=2,
            intent_categories=frozenset(),
        )
        prompt = build_system_prompt(
            goal="fix the bug", classification=classification,
        )
        assert "Document Creation Protocol" not in prompt

    def test_missing_classification_protective_includes_both(self):
        # Missing classification keeps both workflows available.
        prompt = build_system_prompt(goal="fix the bug")
        assert "Email Reading Workflow" in prompt
        assert "Document Creation Protocol" in prompt

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

    @pytest.mark.parametrize("goal_type", ["chat", "web", "artifact"])
    def test_non_coding_followup_does_not_resume_a_software_project(self, goal_type):
        from rune.agent.prompts import PROMPT_CONTINUATION, PROMPT_FOLLOWUP

        cls = ClassificationResult(goal_type=goal_type, confidence=.99, tier=2, is_continuation=True)
        prompt = build_system_prompt(goal="What about today?", goal_category=goal_type, classification=cls)
        assert PROMPT_FOLLOWUP in prompt
        assert PROMPT_CONTINUATION not in prompt
        assert "Code Intelligence" not in prompt

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
        assert "immediately use file_edit" not in prompt
        assert "Execution alone does not authorize" in prompt

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


@pytest.mark.parametrize("category,lookup,deep", [
    ("web", False, False), ("web", True, False), ("web", False, True), ("chat", False, False),
])
def test_source_caveat_policy_matches_normal_and_reserved_answers(category, lookup, deep):
    from rune.agent.litellm_adapter import _answer_request
    from rune.agent.prompts import SOURCE_GAP_GUIDANCE

    classification = ClassificationResult(category, .99, 2, needs_web_evidence=True,
                                         needs_current_source=True, lookup_only=lookup, browser_effect="view")
    prompt = build_system_prompt("Read the current notices", classification, goal_category=category, is_deep_research=deep)
    request = _answer_request({"messages": [{"role": "system", "content": prompt}]}, [], "budget")
    assert "Omit unresolved numbers from search excerpts, even in caveats" in prompt
    assert prompt.count(SOURCE_GAP_GUIDANCE) == 1
    assert SOURCE_GAP_GUIDANCE in request["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_web_loop_keeps_lookup_guidance_and_live_state_out_of_the_cached_prefix(tmp_path, monkeypatch):
    from rune.agent.loop import NativeAgentLoop
    from rune.agent.prompts import SYSTEM_CACHE_BOUNDARY

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    loop = NativeAgentLoop()
    monkeypatch.setattr(loop, "_build_skill_context", lambda *args: None)
    classification = ClassificationResult(goal_type="web", confidence=.9, tier=2)
    context = {"workspace_root": str(tmp_path), "browser_state": {
        "status": "open", "url": "https://example.test/first", "title": "First page"}}
    first = await loop._build_system_prompt("Read the price on that page", classification, context)
    context["browser_state"] = {"status": "closed"}
    second = await loop._build_system_prompt("Read the price on that page", classification, context)
    assert PROMPT_WEB_EFFICIENCY in first and PROMPT_WEB_DEEP not in first
    first_prefix, first_tail = first.split(SYSTEM_CACHE_BOUNDARY)
    second_prefix, second_tail = second.split(SYSTEM_CACHE_BOUNDARY)
    assert first_prefix == second_prefix
    assert "https://example.test/first" in first_tail
    assert '"status": "closed"' in second_tail and "https://example.test/first" not in second


@pytest.mark.asyncio
async def test_mixed_document_work_does_not_require_deep_research(tmp_path, monkeypatch):
    from rune.agent.loop import NativeAgentLoop

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    loop = NativeAgentLoop()
    monkeypatch.setattr(loop, "_build_skill_context", lambda *args: None)
    classification = ClassificationResult(goal_type="full", confidence=.9, tier=2,
                                          intent_categories=frozenset({"document"}), output_expectation="file")
    context = {"workspace_root": str(tmp_path)}
    prompt = await loop._build_system_prompt("Create a report and submit it through the API", classification, context)
    assert PROMPT_WEB_DEEP not in prompt and PROMPT_WEB_EFFICIENCY in prompt
    assert PROMPT_DOCUMENT.replace("{cwd}", str(tmp_path)) in prompt
    classification.goal_type = "research"
    research = await loop._build_system_prompt("Compare source documents", classification, context)
    assert PROMPT_WEB_DEEP not in research
    classification.needs_web_evidence = True
    research = await loop._build_system_prompt("Compare these documents with published sources", classification, context)
    assert PROMPT_WEB_DEEP in research


class TestClassificationResultExtendedFields:
    """Classification defaults and explicit overrides."""

    def test_default_values(self):
        cls = ClassificationResult(goal_type="chat", confidence=0.8, tier=1)
        assert cls.is_continuation is False
        assert cls.is_complex_coding is False
        assert cls.is_multi_task is False
        assert cls.requires_code is False
        assert cls.requires_execution is False
        assert cls.complexity == "moderate"
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


@pytest.mark.parametrize("goal_type,execute", [("research", False), ("research", True), ("full", False)])
def test_document_analysis_does_not_request_an_unasked_deliverable(goal_type, execute):
    c = ClassificationResult(goal_type, .99, 2, intent_categories=frozenset({"document"}),
                             requires_execution=execute, output_expectation="text")
    prompt = build_system_prompt("Compare the supplied reports in your answer", classification=c)
    assert "Document Creation Protocol" not in prompt
    assert "Read supplied files with document_read" in prompt
    assert "Code Intelligence" not in prompt
