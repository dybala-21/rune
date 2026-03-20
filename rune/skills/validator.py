"""Skill definition validator for RUNE.

Validates skill names, descriptions, and frontmatter against the RUNE
skill guidelines (kebab-case names, length limits, reserved words, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Constants

RESERVED_PREFIXES: list[str] = ["claude", "anthropic"]
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500
MAX_NAME_LENGTH = 50

_KEBAB_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_XML_TAG_RE = re.compile(r"<[^>]+>")

TRIGGER_KEYWORDS: list[str] = [
    "use", "when", "ask", "trigger", "invoke", "request",
]


@dataclass(slots=True)
class ValidationResult:
    """Outcome of a validation check."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Name validation

def validate_name(name: str) -> ValidationResult:
    """Validate a skill name.

    Rules:
    - Must not be empty.
    - Max ``MAX_NAME_LENGTH`` characters.
    - Must be kebab-case (``my-skill-name``).
    - Must not contain reserved prefixes (``claude``, ``anthropic``).
    - Must not start with a digit.

    Warnings:
    - Consecutive hyphens (``--``) are discouraged.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not name:
        errors.append("Name is required")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if len(name) > MAX_NAME_LENGTH:
        errors.append(
            f"Name must be at most {MAX_NAME_LENGTH} characters (got {len(name)})"
        )

    if not _KEBAB_CASE_RE.match(name):
        errors.append("Name must be kebab-case (e.g. my-skill-name)")

    lower_name = name.lower()
    for prefix in RESERVED_PREFIXES:
        if prefix in lower_name:
            errors.append(f'"{prefix}" is reserved and cannot appear in skill names')

    if name and name[0].isdigit():
        errors.append("Name must not start with a digit")

    if "--" in name:
        warnings.append("Consecutive hyphens (--) are discouraged")

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings,
    )


# Description validation

def validate_description(description: str) -> ValidationResult:
    """Validate a skill description.

    Rules:
    - Must not be empty.
    - Max ``MAX_DESCRIPTION_LENGTH`` characters.
    - Must not contain XML tags (security restriction).

    Warnings:
    - Descriptions shorter than 20 characters are flagged.
    - Descriptions without trigger keywords are flagged.
    - Trigger examples without quotes are flagged.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not description:
        errors.append("Description is required")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"Description must be at most {MAX_DESCRIPTION_LENGTH} characters "
            f"(got {len(description)})"
        )

    if len(description) < 20:
        warnings.append("Description is very short; consider adding more detail")

    if _XML_TAG_RE.search(description):
        errors.append("Description must not contain XML tags (security restriction)")

    lower_desc = description.lower()
    has_trigger = any(kw in lower_desc for kw in TRIGGER_KEYWORDS)
    if not has_trigger:
        warnings.append(
            "Description should include a trigger phrase indicating when the "
            "skill activates (e.g. 'use when the user asks to ...')"
        )

    has_quotes = bool(re.search(r"""['"`]""", description))
    if not has_quotes and has_trigger:
        warnings.append(
            "Wrap trigger examples in quotes for clarity "
            '(e.g. "deploy my app")'
        )

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings,
    )


# Frontmatter validation

def validate_frontmatter(frontmatter: dict[str, Any]) -> ValidationResult:
    """Validate a complete skill frontmatter dictionary.

    Expects at minimum ``name`` and ``description`` keys.
    """
    errors: list[str] = []
    warnings: list[str] = []

    name = frontmatter.get("name")
    if not name:
        errors.append("Frontmatter must include 'name'")
    else:
        name_result = validate_name(name)
        errors.extend(name_result.errors)
        warnings.extend(name_result.warnings)

    description = frontmatter.get("description")
    if not description:
        errors.append("Frontmatter must include 'description'")
    else:
        desc_result = validate_description(description)
        errors.extend(desc_result.errors)
        warnings.extend(desc_result.warnings)

    compatibility = frontmatter.get("compatibility", "")
    if compatibility and len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        errors.append(
            f"Compatibility must be at most {MAX_COMPATIBILITY_LENGTH} characters"
        )

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings,
    )


def validate_folder_name_match(
    folder_name: str, skill_name: str,
) -> ValidationResult:
    """Validate that the folder name matches the skill name."""
    errors: list[str] = []
    if folder_name != skill_name:
        errors.append(
            f"Folder name '{folder_name}' does not match skill name '{skill_name}'"
        )
    return ValidationResult(valid=len(errors) == 0, errors=errors)
