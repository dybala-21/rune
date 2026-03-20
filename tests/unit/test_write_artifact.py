"""Tests for rune.agent.write_artifact — artifact classification."""

import pytest

from rune.agent.write_artifact import (
    WriteArtifactOptions,
    classify_write_artifact,
)


@pytest.mark.asyncio
class TestClassifyWriteArtifact:
    async def test_python_code_classified_as_structured(self):
        content = "\n".join([
            "def sum(a: int, b: int) -> int:",
            "    total = a + b",
            "    return total",
        ])
        result = await classify_write_artifact(content)
        assert result.is_structured is True
        assert result.score > 0.5

    async def test_json_classified_as_structured(self):
        content = "\n".join([
            "{",
            '  "name": "rune",',
            '  "scripts": {',
            '    "test": "pytest"',
            "  }",
            "}",
        ])
        result = await classify_write_artifact(content)
        assert result.is_structured is True

    async def test_prose_not_classified_as_structured(self):
        content = "\n".join([
            "In this analysis we examined the root cause of the issue.",
            "The reproduction steps and improvement points are described in paragraph form.",
            "Additional experimental results will be reflected in the next iteration.",
        ])
        result = await classify_write_artifact(content)
        assert result.is_structured is False

    async def test_short_memo_not_structured(self):
        result = await classify_write_artifact("todo: verify later")
        assert result.is_structured is False

    async def test_markdown_with_md_hint_not_structured(self):
        content = "\n".join([
            "# Improvement Summary",
            "",
            "- Root cause analysis",
            "- Next actions",
            "",
            "Verification status: verified",
        ])
        result = await classify_write_artifact(
            content,
            WriteArtifactOptions(path_hint="/repo/README.md"),
        )
        assert result.is_structured is False

    async def test_py_extension_hint_helps_classification(self):
        content = "\n".join([
            "import os",
            "from pathlib import Path",
            "",
            "def main():",
            "    print('hello')",
        ])
        result = await classify_write_artifact(
            content,
            WriteArtifactOptions(path_hint="main.py"),
        )
        assert result.is_structured is True
        assert result.score >= 0.9

    async def test_json_extension_hint(self):
        content = '{"key": "value"}'
        result = await classify_write_artifact(
            content,
            WriteArtifactOptions(path_hint="config.json"),
        )
        assert result.is_structured is True
        assert result.score >= 0.95

    async def test_empty_content_returns_not_structured(self):
        result = await classify_write_artifact("")
        assert result.is_structured is False
        assert result.score == 0.0

    async def test_whitespace_only_returns_not_structured(self):
        result = await classify_write_artifact("   \n  \t  ")
        assert result.is_structured is False

    async def test_txt_extension_hint_not_structured(self):
        content = "Some plain text notes about the project"
        result = await classify_write_artifact(
            content,
            WriteArtifactOptions(path_hint="notes.txt"),
        )
        assert result.is_structured is False

    async def test_code_without_extension_detected(self):
        content = "\n".join([
            "def greet(name):",
            "    return f'Hello, {name}'",
            "",
            "if __name__ == '__main__':",
            "    greet('world')",
        ])
        result = await classify_write_artifact(content)
        assert result.is_structured is True
