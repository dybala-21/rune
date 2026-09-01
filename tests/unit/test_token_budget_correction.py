"""The token budget gets the same correction the tool-round cap already had.

Rounds are floored when the workspace is a real code tree, because the
classifier misses `is_complex_coding` often enough that code work otherwise
lands on the small cap and dies mid-diagnosis. The budget had no such floor, so
a build-and-verify run was given the `code_modify` allowance and ran out before
it could read its own compiler errors and fix them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rune.agent.loop import (
    _BUDGET_BY_INTENT,
    NativeAgentLoop,
    TokenBudget,
    _repo_scale_workspace,
)


class _Loop(NativeAgentLoop):
    """Just the budget state the upgrade path touches."""

    def __init__(self, total: int, writes: int = 0, override: int | None = None) -> None:
        self._token_budget = TokenBudget()
        self._token_budget.total = total
        self._structured_writes = writes
        self._budget_upgraded_for_code = False
        self._config = SimpleNamespace(token_budget_override=override)


class TestTheStructuralFloor:
    def test_a_real_project_tree_is_recognised(self):
        assert _repo_scale_workspace(".") is True

    def test_an_empty_directory_is_not(self, tmp_path):
        (tmp_path / "SPEC.md").write_text("task")
        assert _repo_scale_workspace(str(tmp_path)) is False

    def test_a_marker_alone_is_not_enough(self, tmp_path):
        """A marker plus a handful of files is a toy, not a repo."""
        (tmp_path / "Cargo.toml").write_text("[package]")
        (tmp_path / "main.rs").write_text("fn main(){}")
        assert _repo_scale_workspace(str(tmp_path)) is False


class TestTheEvidenceBasedUpgrade:
    """Greenfield work starts in an empty directory, so the structural floor
    cannot see it at t=0 — the run has to earn the budget by writing code."""

    @pytest.mark.parametrize(
        ("writes", "expected"),
        [(0, 200_000), (2, 200_000), (3, 1_000_000), (9, 1_000_000)],
    )
    def test_it_fires_only_once_the_run_has_written_code(self, writes, expected):
        loop = _Loop(_BUDGET_BY_INTENT["code_modify"], writes=writes)
        loop._maybe_upgrade_budget_for_code_work()
        assert loop._token_budget.total == expected

    def test_a_chat_budget_with_no_code_written_is_untouched(self):
        loop = _Loop(_BUDGET_BY_INTENT["chat"])
        loop._maybe_upgrade_budget_for_code_work()
        assert loop._token_budget.total == _BUDGET_BY_INTENT["chat"]

    def test_an_explicit_override_is_left_alone(self):
        """`/goal` names a number; widening it would ignore the caller."""
        loop = _Loop(300_000, writes=9, override=300_000)
        loop._maybe_upgrade_budget_for_code_work()
        assert loop._token_budget.total == 300_000

    def test_it_never_lowers_a_larger_budget(self):
        loop = _Loop(_BUDGET_BY_INTENT["complex_coding"], writes=9)
        loop._maybe_upgrade_budget_for_code_work()
        assert loop._token_budget.total == _BUDGET_BY_INTENT["complex_coding"]

    def test_it_applies_at_most_once(self):
        loop = _Loop(_BUDGET_BY_INTENT["code_modify"], writes=9)
        loop._maybe_upgrade_budget_for_code_work()
        loop._token_budget.total = 123
        loop._maybe_upgrade_budget_for_code_work()
        assert loop._token_budget.total == 123

    def test_the_hard_stop_moves_with_the_budget(self):
        """The run died at 97% of the small budget; the same usage is now early."""
        loop = _Loop(_BUDGET_BY_INTENT["code_modify"], writes=3)
        loop._token_budget.used = 201_076
        assert loop._token_budget.fraction > 0.97   # would have hard-stopped

        loop._maybe_upgrade_budget_for_code_work()

        assert loop._token_budget.fraction < 0.25   # room to keep fixing
