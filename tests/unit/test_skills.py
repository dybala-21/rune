"""Tests for the skills module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from rune.skills.executor import (
    SkillExecutionContext,
    build_skill_context,
    build_skill_context_for_goal,
    merge_skill_contexts,
    validate_skill_requirements,
)
from rune.skills.matcher import _compute_similarity, _tokenize, match_skills
from rune.skills.registry import SkillRegistry, _parse_skill_file
from rune.skills.types import Skill, SkillMatch


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("run unit tests")
        assert tokens == {"run", "unit", "tests"}

    def test_single_char_filtered(self):
        tokens = _tokenize("a big test")
        assert "a" not in tokens
        assert "big" in tokens

    def test_case_insensitive(self):
        tokens = _tokenize("Run TESTS")
        assert "run" in tokens
        assert "tests" in tokens


class TestComputeSimilarity:
    def test_exact_name_match(self):
        skill = Skill(name="test-runner", description="Run unit tests")
        score = _compute_similarity("test-runner", skill)
        assert score > 0.5

    def test_keyword_overlap(self):
        skill = Skill(name="deploy", description="Deploy the application to production")
        score = _compute_similarity("deploy app", skill)
        assert score > 0.3

    def test_no_match(self):
        skill = Skill(name="deploy", description="Deploy application")
        score = _compute_similarity("zebra migration patterns", skill)
        assert score < 0.3

    def test_empty_query(self):
        skill = Skill(name="test", description="Test stuff")
        assert _compute_similarity("", skill) == 0.0


class TestMatchSkills:
    def test_skill_match(self):
        skill = Skill(name="deploy-app", description="Deploy the application to production")
        match = SkillMatch(skill=skill, score=0.85, reason="test")
        assert match.score == 0.85
        assert match.skill.name == "deploy-app"

    def test_match_skills_ordering(self):
        skills = [
            Skill(name="test-runner", description="Run unit tests for the project"),
            Skill(name="deploy-app", description="Deploy the application to production"),
            Skill(name="code-review", description="Review code changes and provide feedback"),
        ]

        matches = match_skills("run tests", skills)
        assert len(matches) > 0
        # Results should be sorted by score descending
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

        # The test-runner skill should score highest for "run tests"
        assert matches[0].skill.name == "test-runner"

    def test_match_skills_empty_query(self):
        skills = [Skill(name="test", description="test")]
        matches = match_skills("", skills)
        assert matches == []

    def test_match_skills_no_matches(self):
        skills = [
            Skill(name="deploy", description="Deploy application"),
        ]
        matches = match_skills("xyzzy quantum entanglement", skills)
        # Low scores may still appear above the 0.05 threshold due to fuzzy matching
        # but they should be scored low
        for m in matches:
            assert m.score < 0.5


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        skill = Skill(name="my-skill", description="Does something")
        reg.register(skill)
        assert reg.get("my-skill") is skill

    def test_get_missing_returns_none(self):
        reg = SkillRegistry()
        assert reg.get("nonexistent") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", description="A"))
        reg.register(Skill(name="b", description="B"))
        skills = reg.list()
        names = {s.name for s in skills}
        assert names == {"a", "b"}

    def test_unregister(self):
        reg = SkillRegistry()
        reg.register(Skill(name="temp", description="Temporary"))
        reg.unregister("temp")
        assert reg.get("temp") is None

    def test_unregister_missing_no_error(self):
        reg = SkillRegistry()
        reg.unregister("nope")  # should not raise

    def test_search(self):
        reg = SkillRegistry()
        reg.register(Skill(name="test-runner", description="Run unit tests"))
        reg.register(Skill(name="deploy", description="Deploy app"))
        matches = reg.search("run tests")
        assert len(matches) > 0
        assert matches[0].skill.name == "test-runner"


class TestParseSkillFile:
    def test_parse_valid_skill_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("---\nname: my-skill\ndescription: A useful skill\nscope: user\n---\n\nBody content here.\n")
            f.flush()
            skill = _parse_skill_file(Path(f.name))

        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "A useful skill"
        assert skill.scope == "user"
        assert "Body content" in skill.body

    def test_parse_file_without_frontmatter(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("Just body content, no frontmatter.\n")
            f.flush()
            skill = _parse_skill_file(Path(f.name))

        assert skill is not None
        # Name falls back to file stem
        assert skill.description == ""

    def test_parse_nonexistent_file(self):
        skill = _parse_skill_file(Path("/tmp/nonexistent-skill-file.md"))
        assert skill is None

    def test_load_skills_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_path = Path(tmpdir) / "SKILL.md"
            skill_path.write_text(
                "---\nname: dir-skill\ndescription: From dir\n---\n\nBody\n"
            )
            reg = SkillRegistry()
            count = reg.load_skills(tmpdir)
            assert count == 1
            assert reg.get("dir-skill") is not None

    def test_load_skills_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillRegistry()
            count = reg.load_skills(tmpdir)
            assert count == 0

    def test_load_skills_nonexistent_directory(self):
        reg = SkillRegistry()
        count = reg.load_skills("/tmp/nonexistent-dir-for-skills")
        assert count == 0


# ---------------------------------------------------------------------------
# Skill Context Building
# ---------------------------------------------------------------------------

class TestBuildSkillContext:
    def _make_skill(self, **overrides):
        defaults = {
            "name": "deploy-app",
            "description": "Deploy the application to production",
            "body": "1. Build the project\n2. Run tests\n3. Deploy to server",
            "metadata": {
                "compatibility": "linux, macos",
                "requires": {
                    "env": ["AWS_ACCESS_KEY"],
                    "bins": ["docker"],
                    "mcp": ["deploy-service"],
                },
                "scripts": ["deploy.sh", "rollback.sh"],
            },
            "file_path": "/home/user/skills/deploy/SKILL.md",
        }
        defaults.update(overrides)
        return Skill(**defaults)

    def test_returns_execution_context(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert isinstance(ctx, SkillExecutionContext)
        assert ctx.active_skill_name == "deploy-app"
        assert ctx.instructions == skill.body

    def test_formatted_context_contains_header(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert "Active Skill: deploy-app" in ctx.formatted_context
        assert "Deploy the application" in ctx.formatted_context

    def test_formatted_context_contains_compatibility(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert "Compatibility: linux, macos" in ctx.formatted_context

    def test_formatted_context_contains_requirements(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert "AWS_ACCESS_KEY" in ctx.formatted_context
        assert "docker" in ctx.formatted_context
        assert "deploy-service" in ctx.formatted_context

    def test_formatted_context_contains_scripts(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert "deploy.sh" in ctx.formatted_context
        assert "rollback.sh" in ctx.formatted_context

    def test_body_included_by_default(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert "Build the project" in ctx.formatted_context

    def test_body_excluded_when_disabled(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill, include_body=False)
        assert "Build the project" not in ctx.formatted_context
        assert "summary mode" in ctx.formatted_context

    def test_body_truncation(self):
        skill = self._make_skill(body="A" * 500)
        ctx = build_skill_context(skill, max_body_chars=100)
        assert "...(truncated)" in ctx.formatted_context

    def test_no_requirements_section_when_empty(self):
        skill = self._make_skill(metadata={})
        ctx = build_skill_context(skill)
        assert "Requirements:" not in ctx.formatted_context

    def test_metadata_dict(self):
        skill = self._make_skill()
        ctx = build_skill_context(skill)
        assert ctx.metadata["description"] == "Deploy the application to production"
        assert ctx.metadata["compatibility"] == "linux, macos"
        assert "requirements" in ctx.metadata


class TestBuildSkillContextForGoal:
    def test_returns_context_for_matching_skill(self):
        reg = SkillRegistry()
        reg.register(Skill(
            name="test-runner",
            description="Run unit tests for the project",
            body="Run pytest",
        ))
        reg.register(Skill(
            name="deploy",
            description="Deploy application",
            body="Deploy steps",
        ))
        ctx = build_skill_context_for_goal("run unit tests", registry=reg)
        assert ctx is not None
        assert ctx.active_skill_name == "test-runner"

    def test_returns_none_for_no_match(self):
        reg = SkillRegistry()
        # Empty registry
        ctx = build_skill_context_for_goal("do something", registry=reg)
        assert ctx is None


class TestMergeSkillContexts:
    def test_merge_empty(self):
        assert merge_skill_contexts([]) == ""

    def test_merge_multiple(self):
        ctx1 = SkillExecutionContext(
            active_skill_name="a",
            instructions="",
            formatted_context="CTX_A",
        )
        ctx2 = SkillExecutionContext(
            active_skill_name="b",
            instructions="",
            formatted_context="CTX_B",
        )
        merged = merge_skill_contexts([ctx1, ctx2])
        assert "CTX_A" in merged
        assert "CTX_B" in merged


class TestValidateSkillRequirements:
    def test_no_requirements(self):
        skill = Skill(name="basic", description="No reqs", metadata={})
        missing = validate_skill_requirements(skill)
        assert missing == []

    def test_missing_env_var(self):
        skill = Skill(
            name="s",
            description="d",
            metadata={"requires": {"env": ["RUNE_TEST_NONEXISTENT_VAR_XYZ"]}},
        )
        missing = validate_skill_requirements(skill)
        assert any("RUNE_TEST_NONEXISTENT_VAR_XYZ" in m for m in missing)

    def test_present_env_var(self):
        with patch.dict(os.environ, {"RUNE_TEST_PRESENT_VAR": "1"}):
            skill = Skill(
                name="s",
                description="d",
                metadata={"requires": {"env": ["RUNE_TEST_PRESENT_VAR"]}},
            )
            missing = validate_skill_requirements(skill)
            assert not any("RUNE_TEST_PRESENT_VAR" in m for m in missing)

    def test_missing_binary(self):
        skill = Skill(
            name="s",
            description="d",
            metadata={"requires": {"bins": ["nonexistent_binary_xyz_12345"]}},
        )
        missing = validate_skill_requirements(skill)
        assert any("nonexistent_binary_xyz_12345" in m for m in missing)

    def test_present_binary(self):
        # 'python3' should be on PATH
        skill = Skill(
            name="s",
            description="d",
            metadata={"requires": {"bins": ["python3"]}},
        )
        missing = validate_skill_requirements(skill)
        assert not any("python3" in m for m in missing)

    def test_mcp_services_reported(self):
        skill = Skill(
            name="s",
            description="d",
            metadata={"requires": {"mcp": ["some-service"]}},
        )
        missing = validate_skill_requirements(skill)
        assert any("some-service" in m for m in missing)
