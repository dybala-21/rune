"""Which deletions the analyzer calls recursive, and what that costs.

`rm -f cache/blob_*.bin` removes forty named files from one directory. It is
not recursive and names no absolute path, yet it scored as "Recursive
deletion with absolute path" and was denied for want of a sandbox — while
`find cache -name 'blob_*.bin' -delete`, which destroys exactly the same
files, scored zero and ran. Agents met the denial by deleting one file at a
time, the consecutive-call block cut them off part way, and the cleanup
tasks ended with a contiguous tail of survivors and an honest report of
partial work.

The flag that caused it was `-f`: the pattern accepted `-[a-z]*f[a-z]*r?`,
where the trailing `r` is optional, so force alone read as force-recursive.

Nothing here argues for permitting more. A recursive delete is still
recursive, a critical path is still critical — and now a delete of a
critical path counts even when nobody passed `-r`.
"""

from __future__ import annotations

import pytest

from rune.safety.analyzer import analyze_command, classify_rm_rf_risk


class TestForceIsNotRecursive:
    @pytest.mark.parametrize("cmd", [
        "rm -f cache/blob_1.bin",
        "rm -f cache/blob_*.bin",
        "rm -f logs/app_{1..30}.log",
        "cd build && rm -f part_*.o && ls -la",
        "rm build/*.o",
    ])
    def test_a_bounded_delete_is_not_a_recursive_one(self, cmd):
        assert classify_rm_rf_risk(cmd) is None
        assert analyze_command(cmd).risk_score < 30

    def test_the_same_files_by_find_score_the_same(self):
        # These two destroy exactly the same forty files. Whatever the
        # verdict is, it cannot depend on which binary spells it.
        by_rm = analyze_command("rm -f cache/blob_*.bin").risk_score
        by_find = analyze_command(
            'find cache -name "blob_*.bin" -delete').risk_score
        assert by_rm == by_find


class TestRecursionStillCounts:
    @pytest.mark.parametrize("cmd", [
        "rm -rf build",
        "rm -fr build",
        "rm -Rf build",
        "rm -r build",
        "rm --recursive build",
    ])
    def test_a_recursive_delete_is_still_flagged(self, cmd):
        assert classify_rm_rf_risk(cmd) == "high"
        assert analyze_command(cmd).risk_score >= 30

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf /etc",
        "rm -rf ~",
    ])
    def test_the_critical_targets_are_still_critical(self, cmd):
        assert classify_rm_rf_risk(cmd) == "critical"


class TestNonRecursiveCriticalTargets:
    """Force alone no longer means recursive, so the path check has to stand
    on its own — otherwise dropping the flag would drop the protection."""

    @pytest.mark.parametrize("cmd", [
        "rm -f /etc/passwd",
        "rm /etc/passwd",
        "rm -f ~/.ssh/id_rsa",
    ])
    def test_a_critical_path_is_critical_without_the_r(self, cmd):
        assert classify_rm_rf_risk(cmd) == "critical"


class TestAPipeCannotLowerTheRisk:
    @pytest.mark.parametrize("cmd", [
        "rm -rf build",
        "rm -f /etc/passwd",
    ])
    def test_appending_a_pipe_never_helps(self, cmd):
        plain = analyze_command(cmd).risk_score
        piped = analyze_command(cmd + " | cat").risk_score
        assert piped >= plain
        assert classify_rm_rf_risk(cmd + " | cat") == classify_rm_rf_risk(cmd)
