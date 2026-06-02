"""Tests for the shared failure-detection protocol in ``output_prefixes``.

These back the F2 refactor that made ``looks_like_failure_output`` the single
source of truth shared by the tool-output formatter (producer) and the
benchmark batch-stop / fail-streak logic (consumer). The detector keys on the
machine-parseable prefixes emitted by ``_format_tool_output`` rather than
re-deriving success from free text, so it must stay language-neutral.
"""

from __future__ import annotations

import pytest

from rune.capabilities.output_prefixes import (
    BLOCKED_PREFIX,
    DENIED_PREFIX,
    ERROR_PREFIX,
    looks_like_failure_output,
)


class TestLooksLikeFailureOutput:
    def test_error_prefix_after_bash_cmd_segment(self):
        # The [ERROR] marker appears after a leading [cmd: ...] segment, so it
        # must be detected within the head window, not only at string start.
        formatted = "[cmd: cmp -s a b] [exit: 1]\n[ERROR] mismatch"
        assert looks_like_failure_output(formatted) is True

    def test_blocked_prefix(self):
        assert looks_like_failure_output(f"{BLOCKED_PREFIX} edit loop") is True

    def test_denied_prefix(self):
        assert looks_like_failure_output(f"{DENIED_PREFIX} user declined") is True

    def test_error_prefix_constant(self):
        assert looks_like_failure_output(f"{ERROR_PREFIX} boom") is True

    def test_bare_error_word(self):
        assert looks_like_failure_output("Error executing bash_execute: ...") is True

    def test_no_changes_marker_anywhere(self):
        # NO CHANGES DETECTED can appear deep in output (phantom write), so it
        # is checked across the whole string, not just the head window.
        tail = "x" * 5000 + "\nNO CHANGES DETECTED\n"
        assert looks_like_failure_output(tail) is True

    @pytest.mark.parametrize(
        "ok",
        [
            "[cmd: echo ok] [exit: 0]\nall good",
            "SAMPLE_CMP:0\nrows match",
            "wrote file successfully",
            "Path: /app/out.txt\ncontents\n[END: /app/out.txt]",
        ],
    )
    def test_clean_output_not_flagged(self, ok):
        assert looks_like_failure_output(ok) is False
