"""Write artifact classifier - detect and classify generated artifacts.

Ported from src/agent/write-artifact.ts (246 lines) - determines if
written content is structured code/config (vs. plain text/docs) using
parser-based and extension-based heuristics.

classifyWriteArtifact(): Classify content as structured (code/config) or prose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types


@dataclass(slots=True)
class WriteArtifactShape:
    """Classification result for a write artifact."""

    is_structured: bool
    score: float
    syntax_density: float = 0.0
    delimiter_line_ratio: float = 0.0
    indentation_ratio: float = 0.0
    paragraph_ratio: float = 0.0
    plain_sentence_ratio: float = 0.0


@dataclass(slots=True)
class WriteArtifactOptions:
    """Options for artifact classification."""

    path_hint: str | None = None


# Constants

MAX_SAMPLE_CHARS = 48_000

DOC_EXTENSIONS = frozenset([".md", ".markdown", ".txt", ".rst", ".adoc", ".log"])
JSON_EXTENSIONS = frozenset([".json"])
YAML_EXTENSIONS = frozenset([".yml", ".yaml"])
GENERIC_CODE_EXTENSIONS = frozenset([
    ".go", ".rs", ".java", ".kt", ".kts", ".py", ".rb", ".php",
    ".swift", ".scala", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs",
    ".sh", ".bash", ".zsh", ".fish", ".sql",
    ".ts", ".tsx", ".js", ".jsx", ".cts", ".mts", ".cjs", ".mjs",
])
CONFIG_EXTENSIONS = frozenset([".toml", ".ini", ".conf", ".cfg", ".properties"])


# Helpers

def _empty_shape() -> WriteArtifactShape:
    return WriteArtifactShape(is_structured=False, score=0.0)


def _to_shape(is_structured: bool, score: float) -> WriteArtifactShape:
    return WriteArtifactShape(
        is_structured=is_structured,
        score=score,
        syntax_density=score if is_structured else 0.0,
    )


def _normalize_extension(path_hint: str | None) -> str | None:
    """Extract and normalize file extension from path hint."""
    if not path_hint or not isinstance(path_hint, str):
        return None
    _, ext = os.path.splitext(path_hint.strip())
    return ext.lower() if ext else None


def _has_substantive_content(sample: str) -> bool:
    """Check if sample has non-comment, non-empty content."""
    for raw_line in sample.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("//", "/*", "*", "#", "--")):
            continue
        return True
    return False


def _parse_json(sample: str) -> bool:
    """Try to parse sample as JSON object/array."""
    try:
        parsed = json_decode(sample)
        return parsed is not None and isinstance(parsed, (dict, list))
    except Exception:
        return False


def _parse_yaml(sample: str) -> bool:
    """Try to parse sample as YAML."""
    try:
        import yaml

        doc = yaml.safe_load(sample)
        return doc is not None and isinstance(doc, (dict, list))
    except Exception:
        return False


def _parse_python_syntax(sample: str) -> bool:
    """Check if sample is valid Python by attempting compile."""
    try:
        compile(sample, "<artifact>", "exec")
        return _has_substantive_content(sample)
    except SyntaxError:
        return False


# Extension-based classification

def _classify_by_extension(sample: str, extension: str) -> WriteArtifactShape | None:
    """Classify using file extension hint."""
    if extension in DOC_EXTENSIONS:
        return _to_shape(False, 0.05)

    if extension in JSON_EXTENSIONS:
        return _to_shape(_parse_json(sample), 0.98)

    if extension in YAML_EXTENSIONS:
        return _to_shape(_parse_yaml(sample), 0.95)

    if extension == ".py":
        return _to_shape(_parse_python_syntax(sample), 0.94)

    if extension in GENERIC_CODE_EXTENSIONS:
        return _to_shape(_has_substantive_content(sample), 0.9)

    if extension in CONFIG_EXTENSIONS:
        return _to_shape(_has_substantive_content(sample), 0.88)

    return None


def _classify_without_extension(sample: str) -> WriteArtifactShape:
    """Classify content without a file extension hint."""
    if _parse_json(sample):
        return _to_shape(True, 0.94)
    if _parse_python_syntax(sample):
        return _to_shape(True, 0.9)
    if _has_substantive_content(sample):
        # Check for code-like indicators
        lines = sample.split("\n")
        code_indicators = sum(
            1 for line in lines
            if line.strip() and (
                line.strip().startswith(("def ", "class ", "import ", "from ", "if ", "for "))
                or "=" in line
                or line.strip().endswith(("{", "}", ";", ":"))
            )
        )
        ratio = code_indicators / max(len(lines), 1)
        if ratio > 0.3:
            return _to_shape(True, 0.85)
    return _empty_shape()


# Public API

async def classify_write_artifact(
    content: str,
    options: WriteArtifactOptions | None = None,
) -> WriteArtifactShape:
    """Classify written content as structured (code/config) or prose.

    Uses parser-based validation (JSON, YAML, Python) and extension-based
    heuristics to determine if content is a structured artifact.
    """
    opts = options or WriteArtifactOptions()

    sample = content[:MAX_SAMPLE_CHARS].strip()
    if not sample:
        return _empty_shape()

    extension = _normalize_extension(opts.path_hint)
    if extension:
        by_extension = _classify_by_extension(sample, extension)
        if by_extension is not None:
            return by_extension

    return _classify_without_extension(sample)
