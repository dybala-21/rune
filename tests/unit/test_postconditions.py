"""What a request implies about the filesystem, fixed before the work.

Only conditions that can be settled by looking are written down. Prose
criteria are left out on purpose: judging whether an answer "addresses the
request" is exactly where automated graders are least reliable, and a rule
nobody can check is a rule that gets quietly satisfied.
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
    def test_an_output_must_end_up_existing(self, workspace):
        conds = derive({"summary.md": "output"}, workspace)
        assert conds == [Postcondition("summary.md", "produced")]

    def test_an_input_that_is_there_must_survive(self, workspace):
        conds = derive({"orders.csv": "input"}, workspace)
        assert conds == [Postcondition("orders.csv", "present")]

    def test_an_input_that_never_existed_gets_no_condition(self, workspace):
        # That case is the missing-input report; duplicating it here would
        # only produce a second complaint about the same thing.
        assert derive({"BUGREPORT.md": "input"}, workspace) == []

    def test_conditions_are_stable_in_order(self, workspace):
        roles = {"summary.md": "output", "orders.csv": "input"}
        assert derive(roles, workspace) == derive(roles, workspace)

    def test_it_can_be_switched_off(self, workspace, monkeypatch):
        monkeypatch.setenv("RUNE_POSTCONDITIONS", "0")
        assert derive({"summary.md": "output"}, workspace) == []


class TestCheck:
    def test_a_satisfied_run_reports_nothing(self, workspace):
        (workspace / "summary.md").write_text("# totals\n")
        conds = derive({"summary.md": "output", "orders.csv": "input"},
                       workspace)
        assert check(conds, workspace) == []

    def test_a_promised_output_that_never_appeared(self, workspace):
        conds = derive({"discounts.md": "output"}, workspace)
        problems = check(conds, workspace)
        assert problems and "discounts.md" in problems[0]

    def test_an_output_created_empty_does_not_count(self, workspace):
        (workspace / "summary.md").write_text("")
        problems = check(derive({"summary.md": "output"}, workspace), workspace)
        assert problems and "empty" in problems[0]

    def test_an_input_that_disappeared_is_caught(self, workspace):
        conds = derive({"policy.md": "input"}, workspace)
        (workspace / "policy.md").unlink()
        problems = check(conds, workspace)
        assert problems and "gone" in problems[0]

    def test_an_output_written_into_a_subdirectory_still_counts(self, workspace):
        conds = derive({"report.md": "output"}, workspace)
        (workspace / "out").mkdir()
        (workspace / "out" / "report.md").write_text("done\n")
        assert check(conds, workspace) == []

    def test_a_file_moved_to_the_trash_does_not_count_as_present(self, workspace):
        conds = derive({"policy.md": "input"}, workspace)
        trash = workspace / ".rune-trash" / "20260101T000000"
        trash.mkdir(parents=True)
        (workspace / "policy.md").rename(trash / "policy.md")
        problems = check(conds, workspace)
        assert problems and "gone" in problems[0]

    def test_conditions_were_fixed_before_the_work(self, workspace):
        # Deriving after the fact would let a run define its own bar: an
        # output it never made would simply not be asked for.
        conds = derive({"summary.md": "output"}, workspace)
        assert check(conds, workspace)          # nothing written yet
        assert derive({}, workspace) == []      # a later, emptier view
        assert check(conds, workspace)          # the original still binds


def test_the_note_names_every_problem():
    note = unmet_note(["a.md was asked for and does not exist",
                       "b.csv was an input to this task and is gone"])
    assert "a.md" in note and "b.csv" in note
    assert "done" in note.lower()
