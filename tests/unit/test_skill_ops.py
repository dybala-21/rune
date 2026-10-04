"""Check skill creation and promotion against the registry and lifecycle APIs."""
from __future__ import annotations

import pytest

from rune.capabilities.skill_ops import (
    SkillCreateParams,
    SkillPromoteParams,
    skill_create,
    skill_promote,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    from rune.skills import registry as reg
    reg._registry = None  # fresh registry per test
    return tmp_path


async def _create(name="my-skill"):
    return await skill_create(SkillCreateParams(
        name=name, description="use when the user asks to do the thing",
        body="1. do it\n2. verify", scope="user", author="tester"))


class TestCreate:
    async def test_metadata_round_trips_colons_and_newlines(self, home):
        from rune.safety.execution_environment import execution_workspace
        from rune.skills.registry import get_skill_registry

        description = "Review: source data\nThen write the report"
        author = "Team: reports"
        result = await skill_create(SkillCreateParams(name="report-review", description=description, author=author, body="Inspect"))
        assert result.success, result.error
        saved = get_skill_registry(workspace=execution_workspace()).get("report-review")
        assert saved.description == description and saved.author == author

    @pytest.mark.asyncio
    async def test_it_writes_and_hot_loads(self, home):
        r = await _create()
        assert r.success, r.error
        assert (home/"skills"/"my-skill"/"SKILL.md").is_file()
        assert r.metadata["loaded"] is True     # load_skills actually ran

    @pytest.mark.asyncio
    async def test_duplicate_is_refused(self, home):
        await _create()
        r = await _create()
        assert not r.success and "already exists" in r.error


class TestPromote:
    @pytest.mark.asyncio
    async def test_promote_transitions_to_active(self, home):
        await _create("promote-me")
        r = await skill_promote(SkillPromoteParams(name="promote-me", force=False))
        assert r.success, r.error
        assert r.metadata["lifecycle"] == "active"

    @pytest.mark.asyncio
    async def test_missing_skill_reports_not_found(self, home):
        r = await skill_promote(SkillPromoteParams(name="ghost", force=False))
        assert not r.success and "not found" in r.error


async def test_project_skill_creation_and_promotion_stay_in_run_workspace(home, tmp_path, monkeypatch):
    from rune.safety.execution_environment import environment_scope
    from rune.skills.persistence import write_skill_to_disk
    from rune.skills.registry import get_skill_registry
    from rune.skills.types import Skill

    launch, workspace = tmp_path / "server", tmp_path / "project"
    launch.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(launch)
    with environment_scope(str(workspace)):
        result = await skill_create(SkillCreateParams(name="project-review", description="Review", body="Read sources", scope="project"))
        assert result.success
        candidate = Skill(name="scoped-candidate", description="Review", body="Check output", scope="project",
                          metadata={"state": "candidate"})
        path = write_skill_to_disk(candidate)
        assert path and path.startswith(str(workspace))
        result = await skill_promote(SkillPromoteParams(name=candidate.name))
        assert result.success
    assert not (launch / ".rune").exists()
    registry = get_skill_registry(workspace=workspace)
    assert registry.get("project-review") is not None
    assert registry.get(candidate.name).metadata["state"] == "active"
