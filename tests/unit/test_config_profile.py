"""Tests for rune.config.profile — ProfileLoader, USER.md / PERSONA.md parsing."""

from __future__ import annotations

import pytest

from rune.config.profile import (
    ProfileLoader,
    _parse_list_items,
    _parse_markdown_sections,
    _parse_persona_config,
    _parse_user_profile,
    get_profile_loader,
)

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

SAMPLE_USER_MD_KR = """\
# User Profile

## \uae30\ubcf8 \uc815\ubcf4
\uc120\ud638 \uc5b8\uc5b4: \ud55c\uad6d\uc5b4
\uc791\uc5c5 \uc2dc\uac04\ub300: 09:00 - 18:00

## \uae08\uae30 \uc0ac\ud56d
- \uc774\ubaa8\uc9c0 \uc0ac\uc6a9\ud558\uc9c0 \ub9c8\uc138\uc694
- \uc601\uc5b4\ub85c \uc751\ub2f5\ud558\uc9c0 \ub9c8\uc138\uc694

## \uc120\ud638 \uc0ac\ud56d
- \uac04\uacb0\ud55c \uc751\ub2f5
- \ucf54\ub4dc \uba3c\uc800 \uc2e4\ud589

## \ucd5c\uadfc \uad00\uc2ec\uc0ac
- 2026-01-15: TypeScript
- React hooks
"""

SAMPLE_PERSONA_MD_EN = """\
# Persona

## Response Style
- Be concise and direct
- Include technical details

## Language
\uae30\ubcf8: English

## Additional Instructions
- Always check security in code reviews
- Include test code
"""

SAMPLE_PERSONA_NO_EMOJI = """\
# Persona

## Response Style
- Be concise
- No emoji

## Language
\uae30\ubcf8: English
"""


class TestParseMarkdownSections:
    def test_extracts_h2_sections(self):
        md = "# Title\n\n## Section A\ncontent a\n\n## Section B\ncontent b\n"
        sections = _parse_markdown_sections(md)
        assert "Section A" in sections
        assert "Section B" in sections
        assert sections["Section A"].strip() == "content a"

    def test_empty_content(self):
        assert _parse_markdown_sections("") == {}


class TestParseListItems:
    def test_extracts_dash_items(self):
        items = _parse_list_items("- foo\n- bar\nbaz\n")
        assert items == ["foo", "bar"]

    def test_extracts_asterisk_items(self):
        items = _parse_list_items("* first\n* second")
        assert items == ["first", "second"]


# ---------------------------------------------------------------------------
# User Profile parsing
# ---------------------------------------------------------------------------


class TestParseUserProfile:
    def test_korean_user_md(self):
        profile = _parse_user_profile(SAMPLE_USER_MD_KR)
        assert profile.preferred_language == "\ud55c\uad6d\uc5b4"
        assert profile.work_hours is not None
        assert profile.work_hours.start == "09:00"
        assert profile.work_hours.end == "18:00"
        assert len(profile.do_nots) == 2
        assert len(profile.organization_preferences) == 2
        assert len(profile.recent_interests) >= 2

    def test_empty_content_returns_defaults(self):
        profile = _parse_user_profile("")
        assert profile.preferred_language is None
        assert profile.do_nots == []
        assert profile.sections == {}


# ---------------------------------------------------------------------------
# Persona Config parsing
# ---------------------------------------------------------------------------


class TestParsePersonaConfig:
    def test_english_persona_md(self):
        persona = _parse_persona_config(SAMPLE_PERSONA_MD_EN)
        assert len(persona.response_style) == 2
        assert persona.default_language == "English"
        assert len(persona.additional_instructions) == 2

    def test_no_emoji_persona(self):
        persona = _parse_persona_config(SAMPLE_PERSONA_NO_EMOJI)
        assert persona.use_emoji is False

    def test_empty_content_returns_defaults(self):
        persona = _parse_persona_config("")
        assert persona.response_style == []
        assert persona.default_language is None


# ---------------------------------------------------------------------------
# ProfileLoader
# ---------------------------------------------------------------------------


class TestProfileLoader:
    @pytest.mark.asyncio
    async def test_initialize_loads_both_files(self, tmp_path):
        config_dir = tmp_path / ".rune"
        config_dir.mkdir()
        (config_dir / "USER.md").write_text(SAMPLE_USER_MD_KR, encoding="utf-8")
        (config_dir / "PERSONA.md").write_text(SAMPLE_PERSONA_MD_EN, encoding="utf-8")

        loader = ProfileLoader(config_dir)
        await loader.initialize()

        profile = loader.get_user_profile()
        assert profile is not None
        assert profile.preferred_language == "\ud55c\uad6d\uc5b4"

        persona = loader.get_persona_config()
        assert persona is not None
        assert persona.default_language == "English"

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, tmp_path):
        config_dir = tmp_path / ".rune"
        config_dir.mkdir()
        (config_dir / "USER.md").write_text("# empty", encoding="utf-8")
        (config_dir / "PERSONA.md").write_text("# empty", encoding="utf-8")

        loader = ProfileLoader(config_dir)
        await loader.initialize()
        await loader.initialize()
        # Should not raise

    @pytest.mark.asyncio
    async def test_handles_missing_files_gracefully(self, tmp_path):
        config_dir = tmp_path / ".rune"
        config_dir.mkdir()

        loader = ProfileLoader(config_dir)
        await loader.initialize()

        profile = loader.get_user_profile()
        assert profile is not None
        assert profile.raw_content == ""

    @pytest.mark.asyncio
    async def test_build_llm_context(self, tmp_path):
        config_dir = tmp_path / ".rune"
        config_dir.mkdir()
        (config_dir / "USER.md").write_text(SAMPLE_USER_MD_KR, encoding="utf-8")
        (config_dir / "PERSONA.md").write_text(SAMPLE_PERSONA_MD_EN, encoding="utf-8")

        loader = ProfileLoader(config_dir)
        await loader.initialize()

        ctx = loader.build_llm_context()
        assert "User Profile" in ctx
        assert "Response Guidelines" in ctx

    @pytest.mark.asyncio
    async def test_reload_reinitializes(self, tmp_path):
        config_dir = tmp_path / ".rune"
        config_dir.mkdir()
        (config_dir / "USER.md").write_text("# v1", encoding="utf-8")
        (config_dir / "PERSONA.md").write_text("# v1", encoding="utf-8")

        loader = ProfileLoader(config_dir)
        await loader.initialize()

        # Overwrite files
        (config_dir / "USER.md").write_text(SAMPLE_USER_MD_KR, encoding="utf-8")
        await loader.reload()

        profile = loader.get_user_profile()
        assert profile is not None
        assert profile.preferred_language == "\ud55c\uad6d\uc5b4"


class TestGetProfileLoader:
    def test_returns_singleton(self):
        loader1 = get_profile_loader()
        loader2 = get_profile_loader()
        assert loader1 is loader2
