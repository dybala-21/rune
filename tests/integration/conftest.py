"""Shared fixtures for RUNE integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rune.agent.cognitive_cache import SessionToolCache
from rune.agent.completion_gate import (
    ExecutionEvidenceSnapshot,
)
from rune.agent.failover import FailoverManager, LLMProfile
from rune.safety.guardian import Guardian
from rune.skills.registry import SkillRegistry
from rune.types import AgentConfig, CompletionTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> AgentConfig:
    """Minimal agent config for testing."""
    return AgentConfig(
        max_iterations=20,
        timeout_seconds=30,
        model="test-model",
        provider="openai",
    )


@pytest.fixture
def guardian() -> Guardian:
    """Fresh Guardian instance."""
    return Guardian()


@pytest.fixture
def skill_registry() -> SkillRegistry:
    """Empty skill registry."""
    return SkillRegistry()


@pytest.fixture
def cognitive_cache() -> SessionToolCache:
    """Fresh cognitive cache with small capacity for tests."""
    return SessionToolCache(max_entries=10)


@pytest.fixture
def failover_manager() -> FailoverManager:
    """Failover manager with test profiles."""
    profiles = [
        LLMProfile(
            name="primary",
            provider="openai",
            model="gpt-test",
            priority=0,
        ),
        LLMProfile(
            name="secondary",
            provider="anthropic",
            model="claude-test",
            priority=1,
        ),
    ]
    return FailoverManager(profiles=profiles, max_retries=2)


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Mock LLM client that returns canned classification responses."""
    client = AsyncMock()
    client.completion = AsyncMock(return_value={
        "choices": [
            {
                "message": {
                    "content": '{"goal_type": "full", "confidence": 0.9, "reason": "test"}'
                }
            }
        ]
    })
    return client


@pytest.fixture
def evidence_snapshot() -> ExecutionEvidenceSnapshot:
    """Pre-populated evidence snapshot."""
    return ExecutionEvidenceSnapshot(
        reads=3,
        writes=2,
        executions=1,
        file_reads=3,
        unique_file_reads=2,
    )


@pytest.fixture
def successful_trace() -> CompletionTrace:
    """A CompletionTrace that looks successful."""
    return CompletionTrace(
        reason="completed",
        final_step=3,
        total_tokens_used=5000,
        evidence_score=0.9,
    )
