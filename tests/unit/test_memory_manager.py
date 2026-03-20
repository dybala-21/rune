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
