"""Skill selection must not activate unrelated bodies or lose prerequisites."""

import pytest

from rune.capabilities.skill_ops import SkillLookupParams, skill_load
from rune.safety.execution_environment import environment_scope
from rune.skills.discovery import catalog


def write_skill(root, name, description, body, extra=""):
    path = root / ".rune" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\nname: {name}\ndescription: >\n  {description}\n{extra}---\n{body}")
    return path


@pytest.mark.asyncio
async def test_catalog_defers_bodies_and_allows_two_independent_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("rune.skills.matcher._semantic_scores", lambda *a: pytest.fail("Small catalog needs no embedding"))
    write_skill(tmp_path, "review", "Inspect evidence", "Read source revision before editing")
    write_skill(tmp_path, "publish", "Prepare a report", "Reopen the saved report")
    write_skill(tmp_path, "candidate", "Do not use yet", "Hidden", "state: candidate\n")
    text = catalog("Prepare a report", str(tmp_path))
    assert "Inspect evidence" in text and "Prepare a report" in text
    assert "Read source revision" not in text and "candidate" not in text
    with environment_scope(str(tmp_path)):
        for name, expected in (("review", "Read source revision"), ("publish", "Reopen")):
            result = await skill_load(SkillLookupParams(name=name))
            assert result.success and expected in result.output
            assert result.metadata["loaded_skill"] == name
        assert not (await skill_load(SkillLookupParams(name="candidate"))).success


@pytest.mark.asyncio
async def test_prerequisites_survive_lifecycle_persistence(tmp_path, monkeypatch):
    from rune.skills.persistence import persist_skill_state
    from rune.skills.registry import get_skill_registry

    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    write_skill(tmp_path, "requires-bin", "Requires a renderer", "Render", "requires:\n  bins: [rune-nonexistent-renderer]\n")
    skill = get_skill_registry(workspace=tmp_path).get("requires-bin")
    assert persist_skill_state(skill)
    with environment_scope(str(tmp_path)):
        result = await skill_load(SkillLookupParams(name=skill.name))
    assert not result.success and "Missing binary" in result.error


@pytest.mark.asyncio
async def test_reusing_skills_does_not_pay_for_another_distillation(monkeypatch):
    from unittest.mock import AsyncMock

    from rune.agent.loop import NativeAgentLoop
    from rune.types import CompletionTrace

    generate = AsyncMock()
    monkeypatch.setattr("rune.agent.memory_bridge.maybe_generate_skill", generate)
    agent = NativeAgentLoop()
    agent._loaded_skills.add("review")
    assert await agent._maybe_distill_skill("Create the report", CompletionTrace(reason="completed")) is None
    generate.assert_not_awaited()


def test_a_new_run_does_not_inherit_skill_credit_or_suppress_its_learning():
    from rune.agent.loop import NativeAgentLoop

    agent = NativeAgentLoop()
    agent._loaded_skills.add("previous-review")
    agent._injected_skill = ("previous-review", 0.0)
    agent._reset_run_state()
    assert not agent._loaded_skills and agent._injected_skill is None


@pytest.mark.parametrize("goal_type", ["research", "web", "browser", "full"])
def test_document_routing_can_reach_the_skills_it_advertises(goal_type):
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    loop = NativeAgentLoop()
    loop._auto_skill = True
    classification = ClassificationResult(goal_type=goal_type, tier=2, confidence=.9, intent_categories={"document"})
    tools = loop._select_tools(classification)
    assert {"skill_search", "skill_load", "document_preview", "file_write"} <= set(tools)


def test_read_only_analysis_can_inspect_rendered_documents_without_write_intent():
    from rune.agent.goal_classifier import ClassificationResult
    from rune.agent.loop import NativeAgentLoop

    tools = NativeAgentLoop()._select_tools(ClassificationResult(goal_type="research", tier=2, confidence=.9))
    assert "document_preview" in tools and "file_write" not in tools
