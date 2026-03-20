"""Tests for rune.skills.validator — ported from validator.test.ts."""


from rune.skills.validator import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    validate_description,
    validate_folder_name_match,
    validate_frontmatter,
    validate_name,
)


class TestValidateName:
    """Tests for validate_name()."""

    def test_passes_for_valid_kebab_case(self):
        result = validate_name("my-skill")
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_passes_for_single_lowercase_word(self):
        result = validate_name("skill")
        assert result.valid is True
        assert len(result.errors) == 0

    def test_passes_for_kebab_case_with_numbers(self):
        result = validate_name("my-skill-2")
        assert result.valid is True

    def test_fails_for_empty_name(self):
        result = validate_name("")
        assert result.valid is False
        assert "Name is required" in result.errors

    def test_returns_early_for_empty_name(self):
        result = validate_name("")
        assert result.valid is False
        assert len(result.errors) == 1

    def test_fails_for_name_exceeding_max_length(self):
        long_name = "a-" + "b" * 49
        result = validate_name(long_name)
        assert result.valid is False
        assert any(str(MAX_NAME_LENGTH) in e for e in result.errors)

    def test_passes_for_name_at_max_length(self):
        name = "a" + "-b" * 24 + "c"  # 50 chars
        assert len(name) == 50
        result = validate_name(name)
        assert not any(str(MAX_NAME_LENGTH) in e and "at most" in e for e in result.errors)

    def test_fails_for_uppercase(self):
        result = validate_name("MySkill")
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)

    def test_fails_for_underscores(self):
        result = validate_name("my_skill")
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)

    def test_fails_for_spaces(self):
        result = validate_name("my skill")
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)

    def test_fails_for_reserved_prefix_claude(self):
        result = validate_name("claude-skill")
        assert result.valid is False
        assert any('"claude"' in e for e in result.errors)

    def test_fails_for_reserved_prefix_anthropic(self):
        result = validate_name("anthropic-tool")
        assert result.valid is False
        assert any('"anthropic"' in e for e in result.errors)

    def test_fails_for_name_starting_with_digit(self):
        result = validate_name("1-skill")
        assert result.valid is False
        assert any("digit" in e for e in result.errors)

    def test_warns_for_consecutive_hyphens(self):
        result = validate_name("my--skill")
        assert any("--" in w for w in result.warnings)

    def test_fails_for_name_ending_with_hyphen(self):
        result = validate_name("skill-")
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)

    def test_fails_for_name_starting_with_hyphen(self):
        result = validate_name("-skill")
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)


class TestValidateDescription:
    """Tests for validate_description()."""

    def test_passes_for_valid_description_with_trigger(self):
        desc = 'This skill triggers when you "ask for help" with downloads'
        result = validate_description(desc)
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_fails_for_empty_description(self):
        result = validate_description("")
        assert result.valid is False
        assert "Description is required" in result.errors

    def test_returns_early_for_empty_description(self):
        result = validate_description("")
        assert len(result.errors) == 1

    def test_fails_for_description_exceeding_max_length(self):
        long_desc = "a" * (MAX_DESCRIPTION_LENGTH + 1)
        result = validate_description(long_desc)
        assert result.valid is False
        assert any(str(MAX_DESCRIPTION_LENGTH) in e for e in result.errors)

    def test_warns_for_short_description(self):
        result = validate_description("short desc trigger")
        assert any("short" in w.lower() for w in result.warnings)

    def test_fails_for_xml_tags_in_description(self):
        result = validate_description("Use when <script>alert(1)</script> detected")
        assert result.valid is False
        assert any("XML" in e for e in result.errors)

    def test_warns_when_no_trigger_keyword(self):
        result = validate_description("This is a skill that does interesting things with data")
        assert any("trigger" in w.lower() for w in result.warnings)


class TestValidateFrontmatter:
    """Tests for validate_frontmatter()."""

    def test_passes_for_valid_frontmatter(self):
        result = validate_frontmatter({
            "name": "my-skill",
            "description": 'Use when the user asks to "do something"',
        })
        assert result.valid is True

    def test_fails_when_name_missing(self):
        result = validate_frontmatter({"description": "desc"})
        assert result.valid is False
        assert any("name" in e.lower() for e in result.errors)

    def test_fails_when_description_missing(self):
        result = validate_frontmatter({"name": "my-skill"})
        assert result.valid is False
        assert any("description" in e.lower() for e in result.errors)

    def test_validates_name_rules(self):
        result = validate_frontmatter({
            "name": "BAD_NAME",
            "description": 'Use when the user asks to "do something"',
        })
        assert result.valid is False
        assert any("kebab-case" in e for e in result.errors)


class TestValidateFolderNameMatch:
    """Tests for validate_folder_name_match()."""

    def test_passes_when_names_match(self):
        result = validate_folder_name_match("my-skill", "my-skill")
        assert result.valid is True

    def test_fails_when_names_differ(self):
        result = validate_folder_name_match("folder-name", "skill-name")
        assert result.valid is False
        assert any("does not match" in e for e in result.errors)
