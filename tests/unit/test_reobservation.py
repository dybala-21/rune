"""What a bulk command touched, and what is left there afterwards.

Two things have to hold. The first is that ordinary work stays untouched:
reading, testing, single-file edits and copies arm nothing, so a run that
never swept a directory pays nothing for this. The second is that the
listing states the position and asks for nothing — the failure next door is
a run that deletes past its brief, and text urging it on would feed that.
"""

from __future__ import annotations

import pytest

from rune.agent.reobservation import (
    bulk_targets,
    mutation_dir,
    observation_note,
    repeated_mutation_dirs,
)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "build").mkdir()
    for i in range(1, 41):
        (tmp_path / "build" / f"part_{i}.o").write_text("x")
    (tmp_path / "logs").mkdir()
    (tmp_path / "run.py").write_text("print(1)\n")
    return tmp_path


def rel(dirs, base):
    return sorted(str(__import__("pathlib").Path(d).relative_to(base.resolve()))
                  for d in dirs)


class TestWhatArmsIt:
    def test_a_glob_delete_names_its_directory(self, workspace):
        assert rel(bulk_targets("rm build/*.o", workspace), workspace) == ["build"]

    def test_a_recursive_delete_counts_without_a_glob(self, workspace):
        assert bulk_targets("rm -rf build", workspace)

    def test_a_directory_already_gone_falls_back_to_its_parent(self, workspace):
        # `rm -r logs` succeeded, so logs/ is not there to be listed; the
        # place its absence shows is the directory above it.
        (workspace / "logs").rmdir()
        assert rel(bulk_targets("rm -rf logs", workspace), workspace) == ["."]

    def test_find_delete_names_its_starting_point(self, workspace):
        got = bulk_targets("find build -name '*.o' -delete", workspace)
        assert rel(got, workspace) == ["build"]

    def test_a_pipeline_is_read_as_one_command(self, workspace):
        # The half that deletes names no path; the half that names one does
        # not delete. Neither is bulk on its own.
        got = bulk_targets("find build -name '*.o' | xargs rm", workspace)
        assert rel(got, workspace) == ["build"]

    def test_every_directory_in_a_chain_is_kept(self, workspace):
        got = bulk_targets("rm build/*.o && rm logs/*.log", workspace)
        assert rel(got, workspace) == ["build", "logs"]

    def test_brace_expansion_counts(self, workspace):
        # `rm logs/app_{1..30}.log` takes thirty files with no `*` in sight,
        # and runs reach for it precisely when they know the count.
        got = bulk_targets("rm -f logs/app_{1..30}.log", workspace)
        assert rel(got, workspace) == ["logs"]
        assert bulk_targets("rm build/{part_1,part_2}.o", workspace)

    def test_a_glob_move_counts(self, workspace):
        assert rel(bulk_targets("mv build/*.o /tmp/", workspace), workspace) \
            == ["build"]


class TestWhatDoesNot:
    @pytest.mark.parametrize("cmd", [
        "ls build/",
        "cat run.py",
        "python -m pytest tests/",
        "grep -r TODO build/",
        "rm build/part_1.o",          # one named file, and its result is seen
        "cp build/part_1.o /tmp/",
        "git status",
        "",
    ])
    def test_ordinary_work_arms_nothing(self, cmd, workspace):
        assert bulk_targets(cmd, workspace) == set()

    def test_a_search_that_merely_mentions_rm_deletes_nothing(self, workspace):
        # -exec runs the token straight after it. Anything looser reads this
        # as a deletion when it is a search.
        assert bulk_targets("find . -name '*.py' -exec grep rm {} +",
                            workspace) == set()

    def test_the_exec_placeholder_is_not_a_file_list(self, workspace):
        # `{}` stands for one path at a time, so it must not count as many —
        # but the deletion here is real and comes from -exec rm.
        assert bulk_targets("find build -name '*.o' -exec rm {} +", workspace)
        assert bulk_targets("echo {}", workspace) == set()

    def test_a_sweep_outside_the_workspace_is_not_listed(self, workspace):
        assert bulk_targets("rm -rf /tmp/scratch-dir/*", workspace) == set()

    def test_a_failed_parse_does_not_raise(self, workspace):
        assert isinstance(bulk_targets("rm build/*.o 'unclosed", workspace), set)

    def test_it_can_be_switched_off(self, workspace, monkeypatch):
        monkeypatch.setenv("RUNE_REOBSERVE", "0")
        assert bulk_targets("rm build/*.o", workspace) == set()
        assert observation_note([str(workspace / "build")], workspace) == ""


class TestDeletingOneFileAtATime:
    """The route the approval gate leaves open, and the one runs take.

    A piped `rm` needs an approval nobody is there to give, so the sweep
    arrives as a long run of single deletions instead. Twenty-nine of them,
    forty-two files still standing, and a done claim — the same failure by
    another road, and it has to arm the same listing.
    """

    def test_a_removal_is_charged_to_its_directory(self, workspace):
        (workspace / "build" / "part_1.o").unlink()
        got = mutation_dir("build/part_1.o", workspace)
        assert got == str((workspace / "build").resolve())

    def test_a_removal_outside_the_workspace_is_ignored(self, workspace):
        assert mutation_dir("/tmp/elsewhere/x.o", workspace) is None

    def test_an_empty_path_is_ignored(self, workspace):
        assert mutation_dir("", workspace) is None

    def test_a_few_deletions_are_not_a_sweep(self):
        assert repeated_mutation_dirs({"/w/build": 4}) == set()

    def test_a_run_of_them_is(self):
        assert repeated_mutation_dirs({"/w/build": 29, "/w/docs": 1}) \
            == {"/w/build"}

    def test_it_can_be_switched_off(self, workspace, monkeypatch):
        monkeypatch.setenv("RUNE_REOBSERVE", "0")
        assert mutation_dir("build/part_1.o", workspace) is None
        assert repeated_mutation_dirs({"/w/build": 29}) == set()


class TestTheListing:
    def test_it_reports_what_survived(self, workspace):
        note = observation_note([str(workspace / "build")], workspace)
        assert "build/" in note
        assert "part_1.o" in note
        assert "40 entries" in note

    def test_a_cleared_directory_says_so(self, workspace):
        note = observation_note([str(workspace / "logs")], workspace)
        assert "empty" in note

    def test_long_listings_are_capped(self, workspace):
        note = observation_note([str(workspace / "build")], workspace)
        assert note.count("part_") <= 20
        assert "shown" in note

    def test_it_asks_for_nothing(self, workspace):
        # Overreach — deleting past the brief — is the neighbouring failure,
        # and an instruction to keep going is how a listing would cause it.
        note = observation_note([str(workspace / "build")], workspace).lower()
        for word in ("should", "must", "finish", "remove", "delete", "clean"):
            assert word not in note

    def test_a_directory_that_vanished_is_skipped(self, workspace):
        gone = workspace / "nowhere"
        assert observation_note([str(gone)], workspace) == ""

    def test_nothing_to_show_produces_nothing(self, workspace):
        assert observation_note([], workspace) == ""
