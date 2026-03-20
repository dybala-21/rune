"""Built-in test suite for the RUNE skill system.

Validates the skill validator, registry, signing, and writer subsystems.
Run directly with ``python -m rune.skills.test_skills``.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from rune.skills.validator import (
    validate_description,
    validate_folder_name_match,
    validate_frontmatter,
    validate_name,
)

# Test utilities

_passed = 0
_failed = 0


def _test(name: str, fn: Any) -> None:
    global _passed, _failed
    try:
        fn()
        print(f"  PASS  {name}")
        _passed += 1
    except Exception as exc:
        print(f"  FAIL  {name}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        _failed += 1


def _assert(condition: bool, message: str = "") -> None:
    if not condition:
        raise AssertionError(message or "Assertion failed")


def _assert_equal(actual: Any, expected: Any, message: str = "") -> None:
    if actual != expected:
        raise AssertionError(
            message or f"Expected {expected!r}, got {actual!r}"
        )


def _assert_contains(haystack: str, needle: str, message: str = "") -> None:
    if needle not in haystack:
        raise AssertionError(
            message or f"Expected {haystack!r} to contain {needle!r}"
        )


# Validator tests

def test_validator() -> None:
    print("\nValidator Tests\n")

    # -- Name --

    _test("validateName: valid kebab-case", lambda: (
        _assert(validate_name("git-commit").valid),
        _assert_equal(len(validate_name("git-commit").errors), 0),
    ))

    _test("validateName: simple name", lambda: (
        _assert(validate_name("commit").valid),
    ))

    _test("validateName: rejects uppercase", lambda: (
        _assert(not validate_name("Git-Commit").valid),
        _assert_contains(validate_name("Git-Commit").errors[0], "kebab-case"),
    ))

    _test("validateName: rejects reserved word 'claude'", lambda: (
        _assert(not validate_name("claude-helper").valid),
        _assert_contains(validate_name("claude-helper").errors[0], "reserved"),
    ))

    _test("validateName: rejects reserved word 'anthropic'", lambda: (
        _assert(not validate_name("anthropic-tools").valid),
    ))

    _test("validateName: rejects starting with number", lambda: (
        _assert(not validate_name("123-skill").valid),
    ))

    # -- Description --

    _test("validateDescription: valid with triggers", lambda: (
        _assert(
            validate_description(
                'Generates commit messages. Use when the user says "commit please".'
            ).valid
        ),
    ))

    _test("validateDescription: warns if no trigger keywords", lambda: (
        _assert(validate_description("This is a simple description.").valid),
        _assert(len(validate_description("This is a simple description.").warnings) > 0),
    ))

    _test("validateDescription: rejects XML tags", lambda: (
        _assert(not validate_description("This has <script>bad</script> content").valid),
        _assert_contains(
            validate_description("This has <script>bad</script> content").errors[0],
            "XML",
        ),
    ))

    _test("validateDescription: rejects too long (>1024)", lambda: (
        _assert(not validate_description("a" * 1025).valid),
        _assert_contains(validate_description("a" * 1025).errors[0], "1024"),
    ))

    # -- Frontmatter --

    _test("validateFrontmatter: valid complete", lambda: (
        _assert(
            validate_frontmatter({
                "name": "test-skill",
                "description": 'Test skill. Use when the user says "test".',
                "license": "MIT",
                "compatibility": "rune v0.1+",
                "metadata": {"author": "test", "version": "1.0.0"},
            }).valid
        ),
    ))

    _test("validateFrontmatter: rejects missing name", lambda: (
        _assert(
            not validate_frontmatter({"description": "Test description"}).valid
        ),
    ))

    _test("validateFrontmatter: rejects missing description", lambda: (
        _assert(not validate_frontmatter({"name": "test-skill"}).valid),
    ))

    # -- Folder name match --

    _test("validateFolderNameMatch: matching", lambda: (
        _assert(validate_folder_name_match("my-skill", "my-skill").valid),
    ))

    _test("validateFolderNameMatch: mismatch", lambda: (
        _assert(not validate_folder_name_match("folder-a", "skill-b").valid),
    ))


# Signing tests

def test_signing() -> None:
    print("\nSigning Tests\n")

    from rune.skills.signing import sign_skill_payload, verify_skill_signature

    _test("sign and verify round-trip", lambda: (
        (sig := sign_skill_payload(
            name="test-skill",
            description="A test skill",
            body="Do something",
            scope="user",
            author="tester",
            secret="my-secret-key",
        )),
        _assert(sig.startswith("hmac-sha256:")),
        _assert(
            verify_skill_signature(
                name="test-skill",
                description="A test skill",
                body="Do something",
                scope="user",
                author="tester",
                secret="my-secret-key",
                signature=sig,
            )
        ),
    ))

    _test("verify fails with wrong secret", lambda: (
        (sig2 := sign_skill_payload(
            name="test-skill",
            description="A test skill",
            body="Do something",
            scope="user",
            author="tester",
            secret="correct-secret",
        )),
        _assert(
            not verify_skill_signature(
                name="test-skill",
                description="A test skill",
                body="Do something",
                scope="user",
                author="tester",
                secret="wrong-secret",
                signature=sig2,
            )
        ),
    ))

    _test("verify fails with tampered body", lambda: (
        (sig3 := sign_skill_payload(
            name="test-skill",
            description="A test skill",
            body="Original body",
            scope="user",
            author="tester",
            secret="my-secret-key",
        )),
        _assert(
            not verify_skill_signature(
                name="test-skill",
                description="A test skill",
                body="Tampered body",
                scope="user",
                author="tester",
                secret="my-secret-key",
                signature=sig3,
            )
        ),
    ))


# Entry point

def main() -> None:
    global _passed, _failed
    _passed = 0
    _failed = 0

    test_validator()
    test_signing()

    print(f"\n{'=' * 40}")
    print(f"Results: {_passed} passed, {_failed} failed")
    print(f"{'=' * 40}\n")

    if _failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
