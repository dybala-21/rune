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


class TestRuleEviction:
    """Soft-cap eviction: the cap used to hard-freeze ALL future failure-learning
    once reached (nothing decremented the rule count; GC only lowered confidence,
    never deleted). These guard the self-maintaining bounded set that replaced it.
    """

    @staticmethod
    def _meta(*entries):
        # entries: (key, eval_count, confidence)
        m = {}
        for key, ev, conf in entries:
            m[key] = {
                "source": "rule_learner", "eval_count": ev, "confidence": conf,
                "hit_count": 0, "human_key": key.split(":")[-1], "created_at": "2020-01-01T00:00:00+00:00",
            }
        return m

    def _patch_state(self, monkeypatch):
        saved = {}
        monkeypatch.setattr(rule_learner, "save_fact_meta", lambda m: saved.update({"meta": dict(m)}))
        removed = []
        monkeypatch.setattr(rule_learner, "remove_learned_fact", lambda k: removed.append(k) or True)
        return saved, removed

    def test_strength_orders_by_engagement_then_confidence(self):
        from rune.memory.rule_learner import _rule_strength
        engaged = {"eval_count": 1, "confidence": 0.40, "created_at": "2020"}
        unproven_hi = {"eval_count": 0, "confidence": 0.95, "created_at": "2020"}
        # engaged rule must outrank a never-engaged higher-confidence one
        assert _rule_strength(engaged) > _rule_strength(unproven_hi)

    def test_evicts_dead_rule_and_persists(self, monkeypatch):
        from rune.memory.rule_learner import _evict_weakest_rule
        saved, removed = self._patch_state(monkeypatch)
        meta = self._meta(
            ("rule:code_modify:aaa", 3, 0.80),   # proven — protect
            ("rule:code_modify:bbb", 0, 0.35),   # dead — evict target
        )
        evicted = _evict_weakest_rule(meta, incoming_confidence=0.40)
        assert evicted == "rule:code_modify:bbb"
        assert "rule:code_modify:bbb" not in meta        # removed from local meta
        assert "bbb" in removed                          # removed from learned.md
        assert "rule:code_modify:bbb" not in saved["meta"]  # persisted

    def test_protects_proven_rules_returns_none(self, monkeypatch):
        from rune.memory.rule_learner import _evict_weakest_rule
        self._patch_state(monkeypatch)
        # weakest is engaged (eval_count>0) → not evictable
        meta = self._meta(
            ("rule:code_modify:aaa", 2, 0.50),
            ("rule:code_modify:bbb", 1, 0.45),
        )
        assert _evict_weakest_rule(meta, incoming_confidence=0.40) is None
        assert len(meta) == 2  # nothing removed

    def test_does_not_evict_for_weaker_incoming(self, monkeypatch):
        from rune.memory.rule_learner import _evict_weakest_rule
        self._patch_state(monkeypatch)
        # dead rule conf 0.55 > incoming 0.40 → keep it (incoming not better)
        meta = self._meta(("rule:code_modify:bbb", 0, 0.55))
        assert _evict_weakest_rule(meta, incoming_confidence=0.40) is None

    def test_ensure_capacity_under_cap_no_eviction(self, monkeypatch):
        from rune.memory.rule_learner import _ensure_rule_capacity
        self._patch_state(monkeypatch)
        monkeypatch.setattr(rule_learner, "_SOFT_CAP", 5)
        meta = self._meta(("rule:code_modify:aaa", 0, 0.35))
        assert _ensure_rule_capacity(meta, 0.40) is True
        assert len(meta) == 1  # untouched, under cap

    def test_ensure_capacity_at_cap_evicts(self, monkeypatch):
        from rune.memory.rule_learner import _ensure_rule_capacity
        self._patch_state(monkeypatch)
        monkeypatch.setattr(rule_learner, "_SOFT_CAP", 2)
        meta = self._meta(
            ("rule:code_modify:aaa", 5, 0.90),
            ("rule:code_modify:bbb", 0, 0.30),
        )
        assert _ensure_rule_capacity(meta, 0.40) is True
        assert len(meta) == 1 and "rule:code_modify:aaa" in meta

    def test_ensure_capacity_at_cap_all_proven_false(self, monkeypatch):
        from rune.memory.rule_learner import _ensure_rule_capacity
        self._patch_state(monkeypatch)
        monkeypatch.setattr(rule_learner, "_SOFT_CAP", 1)
        meta = self._meta(("rule:code_modify:aaa", 3, 0.80))
        assert _ensure_rule_capacity(meta, 0.40) is False


class TestDecayDeletes:
    def test_subthreshold_rule_is_deleted_not_kept(self, monkeypatch):
        from datetime import UTC, datetime, timedelta
        old = (datetime.now(UTC) - timedelta(days=40)).isoformat()
        meta = {"rule:code_modify:zzz": {
            "source": "rule_learner", "eval_count": 0, "hit_count": 0,
            "confidence": 0.31, "human_key": "zzz", "created_at": old,
        }}
        monkeypatch.setattr(rule_learner, "load_fact_meta", lambda: meta)
        saved = {}
        monkeypatch.setattr(rule_learner, "save_fact_meta", lambda m: saved.update({"meta": dict(m)}))
        removed = []
        monkeypatch.setattr(rule_learner, "remove_learned_fact", lambda k: removed.append(k) or True)
        # 0.31 * 0.9 = 0.279 < 0.30 GC threshold → must DELETE
        decay_unused_rules()
        assert "rule:code_modify:zzz" not in saved["meta"]
        assert "zzz" in removed
