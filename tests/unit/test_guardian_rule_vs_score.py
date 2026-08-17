"""A matched rule reports what that rule saw, not what the command is worth.

Guardian scans a list of bash patterns, keeps the worst match, and used to
return it there and then. The worst match among those rules is not the worst
thing about the command: `rm -rf` is not in that list at all — its risk comes
from the analyzer score — so appending `| head -3` matched "File read via
bash" (low), and low decided. `rm -rf build` went from denied to allowed that
way, and so did `sudo rm -rf /var`. Both readings are now taken and the worse
one wins.

The quote fix belongs to the same family: only trailing quotes were stripped
from an rm target, so `rm -rf '/etc'` resolved to a relative path spelled
`'/etc'` and never reached the check that calls /etc critical.
"""

from __future__ import annotations

import pytest

from rune.capabilities.bash import DEFAULT_ALLOWED_EXECUTABLES
from rune.safety.analyzer import classify_rm_rf_risk
from rune.safety.execution_policy import (
    ExecutionPolicyConfig,
    decide_bash_execution,
)
from rune.safety.guardian import get_guardian, risk_to_number

SUFFIXES = ["| head -3", "| cat", "| tail -1", "| less", "| more"]


def verdict(command: str) -> str:
    g = get_guardian()
    cfg = ExecutionPolicyConfig(
        rollout_mode="balanced",
        allowed_executables=list(DEFAULT_ALLOWED_EXECUTABLES),
    )
    return decide_bash_execution(
        command, g.validate(command), cfg,
        has_sandbox_support=False, interactive_approval=True,
    ).decision


DESTRUCTIVE = [
    "rm -rf build",
    "sudo rm -rf /var",
    "find build -name '*.o' | xargs rm",
]


class TestAReadCannotLaunderADelete:
    @pytest.mark.parametrize("base", DESTRUCTIVE)
    @pytest.mark.parametrize("suffix", SUFFIXES)
    def test_piping_into_a_reader_does_not_permit_it(self, base, suffix):
        assert verdict(base) == "deny"
        assert verdict(f"{base} {suffix}") == "deny"

    @pytest.mark.parametrize("base", DESTRUCTIVE)
    def test_the_risk_level_itself_never_drops(self, base):
        g = get_guardian()
        plain = risk_to_number(g.validate(base).risk_level)
        for suffix in SUFFIXES:
            piped = risk_to_number(g.validate(f"{base} {suffix}").risk_level)
            assert piped >= plain, f"{base} {suffix} read as safer"


class TestOrdinaryReadsAreStillOrdinary:
    @pytest.mark.parametrize("cmd", [
        "cat README.md",
        "head -20 logs/app.log",
        "ls -la | head -3",
        "git status | cat",
    ])
    def test_reading_a_file_is_allowed(self, cmd):
        assert verdict(cmd) == "allow"


class TestTheWorseReadingWins:
    def test_a_rule_still_decides_when_it_is_the_worse_one(self):
        # `kill -9 1` scores nothing but the rule list calls it high, and
        # taking the maximum must not lose that.
        assert get_guardian().validate("kill -9 1").risk_level == "high"

    def test_a_score_decides_when_it_is_the_worse_one(self):
        risk = get_guardian().validate("rm -rf build | head -3").risk_level
        assert risk_to_number(risk) >= risk_to_number("medium")


class TestQuotedTargets:
    @pytest.mark.parametrize("cmd", [
        "rm -rf '/etc'",
        'rm -rf "/etc"',
        "rm -rf '/'",
        'rm -rf "$HOME"',
    ])
    def test_a_quoted_critical_path_is_still_critical(self, cmd):
        assert classify_rm_rf_risk(cmd) == "critical"

    def test_an_ordinary_quoted_target_is_unaffected(self):
        assert classify_rm_rf_risk("rm -rf 'build'") == "high"


class TestRewritingsThatUsedToLaunder:
    """Two ways to spell a command so that nothing matched it at all.

    Found by putting seventeen dangerous commands through pairs of
    rewritings — 20,825 variants — and asking whether any of them read as
    milder than the original. Both hits were normalisation gaps, and both
    turned `rm -rf /etc` into a command the classifier called safe.
    """

    @pytest.mark.parametrize("cmd", [
        "rm \\\n -rf /etc",
        "chmod \\\n -R 777 /",
        "kill \\\n -9 1",
        "crontab \\\n -r",
    ])
    def test_a_line_continuation_hides_nothing(self, cmd):
        # Whitespace used to be collapsed first, leaving `\ ` wedged between
        # the command and its flags where no pattern could see past it.
        assert verdict(cmd) == "deny"

    @pytest.mark.parametrize("cmd", [
        "rm${IFS}-rf${IFS}/etc",
        "crontab${IFS}-r",
        "chmod${IFS}-R${IFS}777${IFS}/",
        "kill${IFS}-9${IFS}1",
    ])
    def test_the_braced_ifs_spelling_is_a_space_too(self, cmd):
        # Only `$IFS` was substituted, so `${IFS}` sailed through.
        assert verdict(cmd) == "deny"

    def test_both_at_once(self):
        assert verdict("rm${IFS}\\\n-rf${IFS}/etc") == "deny"
