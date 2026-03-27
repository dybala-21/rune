"""Tests for syntax_guard — registry-based file validation."""

from __future__ import annotations

from rune.agent.syntax_guard import _VALIDATORS, register, validate


class TestValidate:
    """Core validate() function tests."""

    def test_valid_python(self):
        assert validate("test.py", "x = 1\ndef f(): pass") is None

    def test_invalid_python(self):
        result = validate("test.py", "def f(")
        assert result is not None
        assert "line" in result

    def test_valid_json(self):
        assert validate("data.json", '{"key": "value"}') is None

    def test_invalid_json(self):
        result = validate("data.json", "{bad json}")
        assert result is not None

    def test_valid_toml(self):
        assert validate("config.toml", '[section]\nkey = "val"') is None

    def test_invalid_toml(self):
        result = validate("config.toml", "[unclosed")
        assert result is not None

    def test_valid_yaml(self):
        assert validate("config.yaml", "key: value\nlist:\n  - item") is None

    def test_valid_yml_extension(self):
        assert validate("config.yml", "key: value") is None

    def test_unknown_extension_passes(self):
        assert validate("readme.md", "# broken {{{{") is None

    def test_no_extension_passes(self):
        assert validate("Makefile", "all:\n\techo hi") is None

    def test_empty_content_passes(self):
        assert validate("test.py", "") is None

    def test_large_file_skips(self):
        large = "x = 1\n" * 200_000  # > 500KB
        assert validate("test.py", large) is None


class TestPythonValidator:
    """Python-specific validation."""

    def test_syntax_error_with_line_number(self):
        result = validate("test.py", "x = 1\ndef f(\ny = 2")
        assert result is not None
        assert "line" in result

    def test_indentation_error(self):
        result = validate("test.py", "def f():\nx = 1")
        assert result is not None

    def test_multiline_valid(self):
        code = """
def hello(name):
    return f"Hello, {name}!"

class Foo:
    def bar(self):
        pass
"""
        assert validate("test.py", code) is None


class TestJsonValidator:

    def test_nested_valid(self):
        assert validate("x.json", '{"a": {"b": [1, 2, 3]}}') is None

    def test_trailing_comma(self):
        result = validate("x.json", '{"a": 1,}')
        assert result is not None

    def test_empty_object(self):
        assert validate("x.json", "{}") is None

    def test_empty_array(self):
        assert validate("x.json", "[]") is None


class TestRegistry:
    """Registry mechanism tests."""

    def test_builtin_extensions_registered(self):
        assert ".py" in _VALIDATORS
        assert ".json" in _VALIDATORS
        assert ".toml" in _VALIDATORS
        assert ".yaml" in _VALIDATORS
        assert ".yml" in _VALIDATORS

    def test_custom_register(self):
        @register([".test_ext"])
        def _test_validator(content):
            if "BAD" in content:
                return "found BAD"
            return None

        assert validate("file.test_ext", "good content") is None
        assert validate("file.test_ext", "BAD content") == "found BAD"

        # Cleanup
        _VALIDATORS.pop(".test_ext", None)
