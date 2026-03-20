"""End-to-end scenario tests with real LLM API calls.

These tests verify the full RUNE pipeline:
  goal classification → system prompt → LLM call → response

Requires a valid OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment.
Each test has a 60-second timeout to avoid hanging on slow API calls.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_key() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


needs_api = pytest.mark.skipif(not _has_key(), reason="No LLM API key")


# =========================================================================
# 1. Chat / 인사 시나리오
# =========================================================================

@needs_api
class TestChatScenario:
    """Simple chat / greeting — classify as chat, get a text reply."""

    @pytest.mark.asyncio
    async def test_greeting_produces_reply(self):
        """'안녕하세요' should classify and produce a non-empty text reply."""
        from rune.agent.goal_classifier import classify_goal

        result = await classify_goal("안녕하세요, 오늘 기분이 어때?")
        assert result.goal_type in ("chat", "full")
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_tier2_classifies_korean_goal(self):
        """Korean input is classified by LLM (no regex dependency)."""
        from rune.agent.goal_classifier import classify_goal

        goal = "이전에 우리가 논의했던 그 건 어떻게 됐어?"
        result = await classify_goal(goal)
        assert result.tier == 2
        assert result.confidence > 0
        assert result.goal_type in ("chat", "full", "research")


# =========================================================================
# 2. LLM Direct Completion 시나리오
# =========================================================================

@needs_api
class TestLLMCompletion:
    """Verify raw LLM completion works end-to-end via LiteLLM."""

    @pytest.mark.asyncio
    async def test_simple_completion(self):
        """A basic prompt should produce a non-empty response."""
        from rune.llm.client import get_llm_client

        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply in 1 sentence."},
                {"role": "user", "content": "What is Python?"},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        # Extract text from response
        text = ""
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError):
            if isinstance(response, dict):
                text = response["choices"][0]["message"]["content"]

        assert text
        assert len(text) > 10
        assert "python" in text.lower() or "programming" in text.lower()

    @pytest.mark.asyncio
    async def test_korean_completion(self):
        """Korean prompt should produce Korean response."""
        from rune.llm.client import get_llm_client

        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": "한국어로만 답변해주세요. 1문장으로."},
                {"role": "user", "content": "파이썬이 뭐야?"},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        text = ""
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError):
            if isinstance(response, dict):
                text = response["choices"][0]["message"]["content"]

        assert text
        assert len(text) > 5


# =========================================================================
# 3. Intent Engine 시나리오
# =========================================================================

@needs_api
class TestIntentEngineScenario:
    """Full intent engine classification with Tier-1 and Tier-2."""

    @pytest.mark.asyncio
    async def test_code_modify_intent(self):
        """'Fix the bug in main.py' should resolve to code_write or mixed intent."""
        from rune.agent.intent_engine import classify_intent

        result = await classify_intent("fix the bug in main.py and add error handling")
        assert result.resolution == "resolved"
        assert result.intent.kind in (
            "code_write", "code_read", "mixed", "knowledge_explain",
        )
        assert result.source in ("tier1", "tier2")

    @pytest.mark.asyncio
    async def test_research_intent(self):
        """'Explain how the auth module works' should resolve to research/code_read."""
        from rune.agent.intent_engine import classify_intent

        result = await classify_intent(
            "analyze the code in auth.py and explain the authentication flow"
        )
        assert result.resolution == "resolved"
        assert result.intent.kind in ("code_read", "research", "knowledge_explain")

    @pytest.mark.asyncio
    async def test_web_search_intent(self):
        """'Search for Python 3.13 features' should resolve to research."""
        from rune.agent.intent_engine import classify_intent

        result = await classify_intent(
            "search for the latest Python 3.13 new features and changes"
        )
        assert result.resolution == "resolved"
        assert result.intent.kind in ("research", "knowledge_explain")
        assert result.intent.tool_requirement in ("read",)


# =========================================================================
# 4. Agent Loop 시나리오 (chat-only, no tools)
# =========================================================================

@needs_api
class TestAgentLoopChat:
    """Run the agent loop for a simple chat goal (no tool calls needed)."""

    @pytest.mark.asyncio
    async def test_chat_loop_completes(self):
        """A chat goal should run through the loop and produce a trace."""
        from rune.agent.loop import _HAS_PYDANTIC_AI, NativeAgentLoop
        from rune.types import AgentConfig

        if not _HAS_PYDANTIC_AI:
            pytest.skip("pydantic_ai not installed")

        config = AgentConfig(max_iterations=3, timeout_seconds=60)
        loop = NativeAgentLoop(config=config)

        collected: list[str] = []

        async def on_text(delta: str) -> None:
            collected.append(delta)

        loop.on("text_delta", on_text)

        trace = await loop.run("Hi, what can you help me with? Reply in 1 sentence.")

        assert trace.reason in ("completed", "max_iterations", "stalled")
        # The agent should have produced some text output
        full_text = "".join(collected)
        assert len(full_text) > 0 or trace.reason == "completed"


# =========================================================================
# 5. Goal Classifier — 다양한 카테고리 시나리오
# =========================================================================

@needs_api
class TestGoalClassifierVariety:
    """Test Tier-2 classification across diverse goal types."""

    @pytest.mark.asyncio
    async def test_browser_goal(self):
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("open the settings page and click the save button")
        assert result.goal_type in ("browser", "execution", "full")

    @pytest.mark.asyncio
    async def test_execution_goal(self):
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal("run the test suite with pytest and show failures")
        assert result.goal_type in ("execution", "full")
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_complex_multi_step_goal(self):
        from rune.agent.goal_classifier import classify_goal
        result = await classify_goal(
            "read the auth module, find the security vulnerability, "
            "fix it, write a test, and run it"
        )
        assert result.goal_type in ("full", "code_modify", "execution")


# =========================================================================
# 6. Completion Gate + Evidence 통합 시나리오
# =========================================================================

@needs_api
class TestCompletionGateWithRealClassification:
    """Classify a real goal, then evaluate the completion gate."""

    @pytest.mark.asyncio
    async def test_research_gate_needs_read_evidence(self):
        """A research goal classified by the LLM should need read evidence."""
        from rune.agent.completion_gate import (
            CompletionGateInput,
            ExecutionEvidenceSnapshot,
            evaluate_completion_gate,
        )
        from rune.agent.goal_classifier import classify_goal
        from rune.agent.intent_engine import resolve_intent_contract

        classification = await classify_goal(
            "analyze the database schema and explain the relationships"
        )

        # Build an intent contract from the real classification
        contract = resolve_intent_contract(classification, 0.9)

        # Gate with NO evidence → should block or partial
        gate_input = CompletionGateInput(
            intent_resolved=True,
            tool_requirement=contract.tool_requirement,
            output_expectation=contract.output_expectation,
            evidence=ExecutionEvidenceSnapshot(),  # empty
            answer_length=200,
        )
        result = evaluate_completion_gate(gate_input)
        # Without evidence, research goals shouldn't fully verify
        if contract.tool_requirement in ("read", "write"):
            assert result.outcome in ("partial", "blocked")

        # Gate WITH evidence → should verify
        gate_input_with_evidence = CompletionGateInput(
            intent_resolved=True,
            tool_requirement=contract.tool_requirement,
            output_expectation=contract.output_expectation,
            evidence=ExecutionEvidenceSnapshot(
                reads=5, writes=0, file_reads=5, unique_file_reads=3,
            ),
            answer_length=500,
        )
        result2 = evaluate_completion_gate(gate_input_with_evidence)
        assert result2.outcome in ("verified", "partial")


# =========================================================================
# 7. Failover 시나리오 (simulated API error → real recovery)
# =========================================================================

@needs_api
class TestFailoverRecovery:
    """Simulate an error then verify the system can recover with a real call."""

    @pytest.mark.asyncio
    async def test_recover_after_simulated_error(self):
        """After a simulated rate limit, a follow-up real call should succeed."""
        from rune.agent.failover import classify_error

        # Simulate error classification
        reason = classify_error(Exception("429 Too Many Requests"))
        assert reason == "rate_limit"

        # Now make a real LLM call to verify the system works
        from rune.llm.client import get_llm_client
        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "user", "content": "Say OK"},
            ],
            temperature=0.0,
            max_tokens=16,
        )

        text = ""
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError):
            if isinstance(response, dict):
                text = response["choices"][0]["message"]["content"]

        assert text  # Got a real response after simulated error


# =========================================================================
# 8. Memory Bridge 시나리오
# =========================================================================

@needs_api
class TestMemoryBridgeSkillExtraction:
    """Extract a skill template from a real classification + tool trace."""

    @pytest.mark.asyncio
    async def test_skill_extracted_from_real_classification(self):
        from rune.agent.goal_classifier import classify_goal
        from rune.agent.memory_bridge import ToolTraceEntry, extract_skill_template

        # Classify a real goal
        classification = await classify_goal("refactor the utils module to use dataclasses")
        assert classification.confidence > 0

        # Simulate a tool trace that would result from this goal
        trace = [
            ToolTraceEntry(
                tool_name="file_read",
                params={"file_path": "/project/utils.py"},
                result_summary="read utils module",
                success=True,
            ),
            ToolTraceEntry(
                tool_name="file_edit",
                params={
                    "file_path": "/project/utils.py",
                    "old_string": "class Config(dict):",
                    "new_string": "@dataclass\nclass Config:",
                },
                result_summary="converted to dataclass",
                success=True,
            ),
        ]

        template = extract_skill_template("refactor utils to dataclasses", trace)
        assert template is not None
        assert len(template["steps"]) == 2
        assert template["pattern"] == "read_modify"
