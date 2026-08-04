"""What a request implies about the filesystem, fixed before the work.

Only conditions that can be settled by looking are written down. Prose
criteria are left out on purpose: judging whether an answer "addresses the
request" is exactly where automated graders are least reliable, and a rule
nobody can check is a rule that gets quietly satisfied.

An output the request named used to earn a condition of its own. It does
not any more, and the tests below say so directly, because the rule reads
as obviously correct until you meet a task whose honest answer is that the
output cannot be produced.
"""

from __future__ import annotations

import pytest

from rune.agent.postconditions import Postcondition, check, derive, unmet_note


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "orders.csv").write_text("id,total\n1,10\n")
    (tmp_path / "policy.md").write_text("rules\n")
    return tmp_path


class TestDerive:
    def test_a_named_output_carries_no_obligation(self, workspace):
        # Whether the request is owed its output depends on what the data
        # supports, which no amount of looking at the filesystem settles.
        # Asked for a discount report from a table with no discount column,
        # the right answer is that it cannot be produced — and a rule
        # demanding the file appear argues for inventing one.
        assert derive({"discounts.md": "output"}, workspace) == []

    def test_an_input_that_is_there_must_survive(self, workspace):
        conds = derive({"orders.csv": "input"}, workspace)
        assert conds == [Postcondition("orders.csv", "present")]

    def test_an_input_that_never_existed_gets_no_condition(self, workspace):
        # That case is the missing-input report; duplicating it here would
        # only produce a second complaint about the same thing.
        assert derive({"BUGREPORT.md": "input"}, workspace) == []

    def test_conditions_are_stable_in_order(self, workspace):
        roles = {"policy.md": "input", "orders.csv": "input"}
        assert derive(roles, workspace) == derive(roles, workspace)

    def test_it_can_be_switched_off(self, workspace, monkeypatch):
        monkeypatch.setenv("RUNE_POSTCONDITIONS", "0")
        assert derive({"orders.csv": "input"}, workspace) == []


class TestCheck:
    def test_a_satisfied_run_reports_nothing(self, workspace):
        conds = derive({"policy.md": "input", "orders.csv": "input"},
                       workspace)
        assert check(conds, workspace) == []

    def test_an_output_that_never_appeared_is_not_complained_about(self, workspace):
        # The run that says "there is no discount data" is the correct one.
        conds = derive({"discounts.md": "output"}, workspace)
        assert check(conds, workspace) == []

    def test_an_input_that_disappeared_is_caught(self, workspace):
        conds = derive({"policy.md": "input"}, workspace)
        (workspace / "policy.md").unlink()
        problems = check(conds, workspace)
        assert problems and "gone" in problems[0]

    def test_an_input_moved_into_a_subdirectory_still_counts(self, workspace):
        conds = derive({"policy.md": "input"}, workspace)
        (workspace / "out").mkdir()
        (workspace / "policy.md").rename(workspace / "out" / "policy.md")
        assert check(conds, workspace) == []

    def test_a_file_moved_to_the_trash_does_not_count_as_present(self, workspace):
        conds = derive({"policy.md": "input"}, workspace)
        trash = workspace / ".rune-trash" / "20260101T000000"
        trash.mkdir(parents=True)
        (workspace / "policy.md").rename(trash / "policy.md")
        problems = check(conds, workspace)
        assert problems and "gone" in problems[0]

    def test_conditions_were_fixed_before_the_work(self, workspace):
        # Deriving after the fact would let a run define its own bar: a file
        # it had already deleted would simply stop being asked about.
        conds = derive({"policy.md": "input"}, workspace)
        (workspace / "policy.md").unlink()
        assert check(conds, workspace)          # the loss is reported
        assert derive({}, workspace) == []      # a later, emptier view
        assert check(conds, workspace)          # the original still binds


def test_the_note_names_every_problem():
    note = unmet_note(["a.csv was an input to this task and is gone",
                       "b.csv was an input to this task and is gone"])
    assert "a.csv" in note and "b.csv" in note
    assert "done" in note.lower()
