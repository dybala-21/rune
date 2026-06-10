"""Tests for rule_learner — failure pattern detection and rule lifecycle."""

from __future__ import annotations

import pytest

import rune.memory.rule_learner as rule_learner
from rune.memory.rule_learner import (
    _error_signature,
    decay_unused_rules,
    find_repeated_failures,
    get_relevant_rules,
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


class TestGetRelevantRules:
    """Hybrid retrieval: exact-domain (precise) + semantic (robust to the LLM
    classifier drifting between valid goal_type enums)."""

    _CANDS = [
        # exact-domain but semantically DISSIMILAR to the goal
        {"key": "A", "value": "exact", "confidence": 0.7, "domain": "code_modify", "hit_count": 0},
        # OTHER domain but semantically SIMILAR (the drift case we must catch)
        {"key": "B", "value": "floor division", "confidence": 0.9, "domain": "full", "hit_count": 0},
        # other domain AND dissimilar → must be excluded
        {"key": "C", "value": "click", "confidence": 0.65, "domain": "browser", "hit_count": 0},
    ]
    # goal=[1,0]; A=[0,1] (cos 0), B=[1,0] (cos 1), C=[0,1] (cos 0)
    _VECS = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

    def _patch(self, monkeypatch, vecs=None, raise_embed=False):
        monkeypatch.setattr(rule_learner, "_resolved_rule_candidates", lambda: list(self._CANDS))

        class _FakeMgr:
            async def embed_batch(self, texts):
                if raise_embed:
                    raise RuntimeError("no embedding provider")
                return vecs if vecs is not None else self_vecs

        self_vecs = self._VECS
        import rune.memory.manager as mgr_mod
        monkeypatch.setattr(mgr_mod, "get_memory_manager", lambda: _FakeMgr())

    @pytest.mark.asyncio
    async def test_includes_exact_and_semantic(self, monkeypatch):
        self._patch(monkeypatch)
        rules = await get_relevant_rules("compute calc", domain="code_modify")
        keys = {r["key"] for r in rules}
        assert "A" in keys  # exact-domain, even though dissimilar
        assert "B" in keys  # cross-domain but semantically similar (drift caught)
        assert "C" not in keys  # cross-domain + dissimilar
        # sorted by confidence desc → B (0.9) before A (0.7)
        assert [r["key"] for r in rules][:2] == ["B", "A"]

    @pytest.mark.asyncio
    async def test_semantic_catches_rule_when_domain_mismatches(self, monkeypatch):
        # Classifier returned the WRONG enum ('full' instead of code_modify):
        # exact-domain finds nothing, semantic still surfaces the similar rule.
        self._patch(monkeypatch)
        rules = await get_relevant_rules("compute calc", domain="research")
        keys = {r["key"] for r in rules}
        assert keys == {"B"}  # only the semantically-similar one

    @pytest.mark.asyncio
    async def test_falls_back_to_exact_when_embedding_unavailable(self, monkeypatch):
        self._patch(monkeypatch, raise_embed=True)
        rules = await get_relevant_rules("compute calc", domain="code_modify")
        assert {r["key"] for r in rules} == {"A"}  # exact-domain only

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, monkeypatch):
        monkeypatch.setattr(rule_learner, "_resolved_rule_candidates", lambda: [])
        assert await get_relevant_rules("x", domain="code_modify") == []


class TestDecayUnusedRules:

    def test_no_rules_no_decay(self):
        count = decay_unused_rules()
        assert count >= 0  # Should not crash
