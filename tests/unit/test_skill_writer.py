"""Tests for rune.skills.writer — ported from writer.test.ts."""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from rune.skills.signing import verify_skill_signature
from rune.skills.writer import WriteSkillInput, write_skill_file


@pytest.fixture
def tmp_base(tmp_path):
    return str(tmp_path)


def _parse_skill_markdown(content: str):
    """Parse frontmatter + body from SKILL.md content."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML required for writer tests")
    m = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", content)
    assert m is not None, "Invalid skill markdown format"
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()
    return fm, body


class TestWriteSkillFile:
    """Tests for write_skill_file()."""

    def test_writes_skill_md_to_correct_path(self, tmp_base):
        inp = WriteSkillInput(
            name="writer-basic",
            description="writer test description",
            body="# writer body",
            scope="user",
            author="rune-agent",
            base_dir=tmp_base,
            metadata={"auto_generated": True},
        )
        result = write_skill_file(inp)
        expected_path = os.path.join(tmp_base, "writer-basic", "SKILL.md")
        assert result.skill_md_path == expected_path
        assert Path(expected_path).exists()

    def test_auto_signs_skill(self, tmp_base):
        inp = WriteSkillInput(
            name="writer-basic",
            description="writer test description",
            body="# writer body",
            scope="user",
            author="rune-agent",
            base_dir=tmp_base,
        )
        result = write_skill_file(inp)
        assert result.signed is True

    def test_frontmatter_contains_name_and_metadata(self, tmp_base):
        inp = WriteSkillInput(
            name="writer-fm",
            description="fm test",
            body="# body",
            scope="user",
            author="rune-agent",
            base_dir=tmp_base,
            metadata={"auto_generated": True},
        )
        result = write_skill_file(inp)
        content = Path(result.skill_md_path).read_text()
        fm, body = _parse_skill_markdown(content)

        assert fm["name"] == "writer-fm"
        metadata = fm.get("metadata", {})
        assert metadata["author"] == "rune-agent"
        assert metadata["auto_generated"] is True

    def test_explicit_signing_key_produces_verifiable_signature(self, tmp_base):
        with patch.dict(os.environ, {"RUNE_SKILL_SIGNING_KEY": "unit-test-secret"}):
            inp = WriteSkillInput(
                name="writer-signed",
                description="line1\nline2",
                body="# writer body\n\ncontent",
                scope="user",
                author="rune-agent",
                base_dir=tmp_base,
            )
            result = write_skill_file(inp)
            assert result.signed is True
            assert result.signature is not None
            assert "hmac-sha256:" in result.signature

            content = Path(result.skill_md_path).read_text()
            fm, body = _parse_skill_markdown(content)
            metadata = fm.get("metadata", {})
            signature = str(metadata.get("signature", ""))
            assert "hmac-sha256:" in signature

            assert verify_skill_signature(
                name=str(fm["name"]),
                description=str(fm["description"]),
                body=body,
                scope="user",
                author="rune-agent",
                secret="unit-test-secret",
                signature=signature,
            ) is True

    def test_creates_skill_directory(self, tmp_base):
        inp = WriteSkillInput(
            name="new-dir-skill",
            description="test",
            body="body",
            base_dir=tmp_base,
        )
        result = write_skill_file(inp)
        assert Path(result.skill_dir).is_dir()

    def test_result_contains_scope_and_author(self, tmp_base):
        inp = WriteSkillInput(
            name="meta-check",
            description="d",
            body="b",
            scope="project",
            author="tester",
            base_dir=tmp_base,
        )
        result = write_skill_file(inp)
        assert result.scope == "project"
        assert result.author == "tester"
