"""Tests for MemoryManager — calculate_importance() and add_safety_rule()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune.memory.manager import MemoryManager
from rune.memory.store import MemoryStore


@pytest.fixture
def mem_store(tmp_dir):
    """Create a real MemoryStore backed by a temp DB."""
    store = MemoryStore(db_path=tmp_dir / "test_memory.db")
    yield store
    store.close()


@pytest.fixture
def manager(mem_store):
    """Create a MemoryManager with a real store but mocked vector store."""
    vec_mock = MagicMock()
    mgr = MemoryManager(store=mem_store, vector_store=vec_mock)
    return mgr


class TestCalculateImportance:
    def test_base_score(self, manager):
        """Minimal result gives base score of 0.3."""
        score = manager.calculate_importance({})
        assert score == pytest.approx(0.3)

    def test_complexity_bonus_5_steps(self, manager):
        score = manager.calculate_importance({"steps": 5})
        assert score == pytest.approx(0.4)

    def test_complexity_bonus_10_steps(self, manager):
        score = manager.calculate_importance({"steps": 10})
        assert score == pytest.approx(0.5)

    def test_complexity_bonus_20_steps(self, manager):
        score = manager.calculate_importance({"steps": 20})
        assert score == pytest.approx(0.6)

    def test_complexity_bonus_below_5_steps(self, manager):
        score = manager.calculate_importance({"steps": 3})
        assert score == pytest.approx(0.3)

    def test_lessons_bonus(self, manager):
        score = manager.calculate_importance({}, lessons=["lesson1", "lesson2"])
        # 0.3 base + 0.2 lessons
        assert score == pytest.approx(0.5)

    def test_lessons_bonus_capped(self, manager):
        score = manager.calculate_importance({}, lessons=["a", "b", "c", "d", "e"])
        # 0.3 base + min(0.3, 5*0.1) = 0.3 + 0.3 = 0.6
        assert score == pytest.approx(0.6)

    def test_failure_bonus(self, manager):
        score = manager.calculate_importance({"success": False})
        # 0.3 base + 0.1 failure
        assert score == pytest.approx(0.4)

    def test_success_no_failure_bonus(self, manager):
        score = manager.calculate_importance({"success": True})
        assert score == pytest.approx(0.3)

    def test_outputs_bonus_few_files(self, manager):
        score = manager.calculate_importance({"changed_files": ["a.py"]})
        # 0.3 base + 0.1 outputs
        assert score == pytest.approx(0.4)

    def test_outputs_bonus_many_files(self, manager):
        score = manager.calculate_importance({"changed_files": ["a", "b", "c", "d", "e"]})
        # 0.3 base + 0.2 outputs
        assert score == pytest.approx(0.5)

    def test_outputs_bonus_no_files(self, manager):
        score = manager.calculate_importance({"changed_files": []})
        assert score == pytest.approx(0.3)

    def test_combined_max_capped_at_1(self, manager):
        # All bonuses: 0.3 + 0.3(steps>=20) + 0.3(lessons) + 0.1(failure) + 0.2(files>=5) = 1.2 -> capped at 1.0
        score = manager.calculate_importance(
            {"steps": 25, "success": False, "changed_files": list(range(10))},
            lessons=["a", "b", "c", "d"],
        )
        assert score == pytest.approx(1.0)

    def test_combined_moderate(self, manager):
        # 0.3 + 0.2(10 steps) + 0.1(1 lesson) + 0.1(1 file) = 0.7
        score = manager.calculate_importance(
            {"steps": 10, "changed_files": ["x.py"]},
            lessons=["learned something"],
        )
        assert score == pytest.approx(0.7)


class TestAddSafetyRule:
    @pytest.mark.asyncio
    async def test_adds_rule_to_working_memory(self, manager, mem_store):
        await manager.add_safety_rule("path_block", "/etc/passwd", reason="sensitive file")

        rules = manager.working.safety_rules
        assert len(rules) == 1
        assert rules[0]["type"] == "path_block"
        assert rules[0]["pattern"] == "/etc/passwd"
        assert rules[0]["reason"] == "sensitive file"

    @pytest.mark.asyncio
    async def test_multiple_rules(self, manager):
        await manager.add_safety_rule("cmd_block", "rm -rf /", reason="dangerous")
        await manager.add_safety_rule("path_block", "/root", reason="root home")

        assert len(manager.working.safety_rules) == 2


class TestRankDurableFacts:
    """Durable-fact injection ranking (mirrors Hermes' always-on USER.md).

    Regression guard for the flat ``durable_facts[:10]`` parse-order slice that
    buried a just-saved 'allergic to peanuts' preference at rank 15/16 even
    though the context was well under its char budget.
    """

    @staticmethod
    def _fact(category, key, value):
        from rune.memory.tiered_memory import DurableMemoryEntry

        return DurableMemoryEntry(category=category, key=key, value=value, confidence=1.0)

    def test_recent_preference_survives_beyond_old_count_cap(self):
        from rune.memory.manager import _rank_durable_facts

        # 15 older prefs + 1 just-saved allergy at the end (append order).
        facts = [self._fact("preference", f"p{i}", f"old pref {i}") for i in range(15)]
        facts.append(self._fact("preference", "diet", "allergic to peanuts, avoids thai food"))
        ranked = _rank_durable_facts(facts, goal="suggest a cuisine for dinner")
        values = " ".join(f.value for f in ranked)
        assert "peanuts" in values  # would fall off a [:10] slice

    def test_preference_ranked_ahead_of_project(self):
        from rune.memory.manager import _rank_durable_facts

        facts = [
            self._fact("preference", "diet", "allergic to peanuts"),
            self._fact("project", "fmt", "use pptx for slides"),
        ]
        ranked = _rank_durable_facts(facts, goal=None, char_budget=40)
        # Only one fits in 40 chars — must be the personal-profile preference.
        assert ranked[0].category == "preference"

    def test_goal_relevance_orders_within_category(self):
        from rune.memory.manager import _rank_durable_facts

        facts = [
            self._fact("preference", "a", "uses neovim editor"),
            self._fact("preference", "b", "prefers postgres database"),
        ]
        ranked = _rank_durable_facts(facts, goal="connect to the database")
        assert ranked[0].value == "prefers postgres database"

    def test_char_budget_bounds_output(self):
        from rune.memory.manager import _rank_durable_facts

        facts = [self._fact("preference", f"k{i}", "x" * 50) for i in range(100)]
        ranked = _rank_durable_facts(facts, goal=None, char_budget=300)
        assert 0 < len(ranked) < 100

    def test_empty_returns_empty(self):
        from rune.memory.manager import _rank_durable_facts

        assert _rank_durable_facts([], goal="anything") == []
