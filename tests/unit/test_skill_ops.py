"""skill.create and skill.promote must reach real registry/lifecycle methods.

Both capabilities are registered and exposed to the agent, and both called
registry methods that never existed (get_skill, load_skill_from_path,
promote_skill) — an AttributeError at runtime that no test covered. These
pin them to the methods the registry and lifecycle module actually define.
"""
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
