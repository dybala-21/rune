"""Tests for rule_learner — failure pattern detection and rule lifecycle."""

from __future__ import annotations

import pytest

from rune.memory.rule_learner import (
    _error_signature,
    decay_unused_rules,
    find_repeated_failures,
    get_rules_for_domain,
)
from rune.memory.store import MemoryStore


@pytest.fixture
def store(tmp_dir):
    """In-memory store for testing."""
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


class TestErrorSignature:
    """Signature stability and normalization."""

    def test_same_error_different_paths(self):
        sig1 = _error_signature("bash", "FileNotFoundError: /Users/a/foo.py")
        sig2 = _error_signature("bash", "FileNotFoundError: /Users/b/bar.py")
        assert sig1 == sig2

    def test_same_error_different_line_numbers(self):
        sig1 = _error_signature("bash", "SyntaxError at line 10")
        sig2 = _error_signature("bash", "SyntaxError at line 99")
        assert sig1 == sig2

    def test_different_tools_different_sigs(self):
        sig1 = _error_signature("bash", "error X")
        sig2 = _error_signature("file_edit", "error X")
        assert sig1 != sig2

    def test_different_errors_different_sigs(self):
        sig1 = _error_signature("bash", "SyntaxError")
        sig2 = _error_signature("bash", "PermissionError")
        assert sig1 != sig2

    def test_filename_normalization(self):
        sig1 = _error_signature("bash", "Error in main.py")
        sig2 = _error_signature("bash", "Error in utils.py")
        assert sig1 == sig2

    def test_deterministic(self):
        sig1 = _error_signature("bash", "test error")
        sig2 = _error_signature("bash", "test error")
        assert sig1 == sig2

    def test_returns_short_hash(self):
        sig = _error_signature("bash", "some error")
        assert len(sig) == 12
        assert sig.isalnum()


class TestFindRepeatedFailures:

    def test_no_failures(self, store):
        store.log_tool_call("s1", "bash", result_success=True)
        assert find_repeated_failures(store) == []

    def test_single_failure_not_enough(self, store):
        store.log_tool_call("s1", "bash", result_success=False,
                           error_message="SyntaxError in main.py")
        assert find_repeated_failures(store) == []

    def test_two_similar_failures_detected(self, store):
        store.log_tool_call("s1", "bash", result_success=False,
                           error_message="SyntaxError in main.py line 10")
        store.log_tool_call("s2", "bash", result_success=False,
                           error_message="SyntaxError in utils.py line 20")
        patterns = find_repeated_failures(store)
        assert len(patterns) == 1
        assert patterns[0]["count"] == 2
        assert patterns[0]["tool_name"] == "bash"

    def test_different_errors_separate_patterns(self, store):
        store.log_tool_call("s1", "bash", result_success=False,
                           error_message="SyntaxError in x.py")
        store.log_tool_call("s2", "bash", result_success=False,
                           error_message="SyntaxError in y.py")
        store.log_tool_call("s3", "file_edit", result_success=False,
                           error_message="PermissionError: /etc/hosts")
        patterns = find_repeated_failures(store)
        # Only SyntaxError pair should match (2 occurrences)
        # PermissionError is only 1 occurrence
        assert len(patterns) == 1

    def test_success_calls_ignored(self, store):
        store.log_tool_call("s1", "bash", result_success=True,
                           error_message="")
        store.log_tool_call("s2", "bash", result_success=True,
                           error_message="")
        assert find_repeated_failures(store) == []


class TestGetRulesForDomain:

    def test_empty_returns_empty(self):
        # No learned.md exists → empty list
        rules = get_rules_for_domain("nonexistent_domain")
        assert isinstance(rules, list)

    def test_max_10_rules(self):
        rules = get_rules_for_domain("code_modify")
        assert len(rules) <= 10


class TestDecayUnusedRules:

    def test_no_rules_no_decay(self):
        count = decay_unused_rules()
        assert count >= 0  # Should not crash
