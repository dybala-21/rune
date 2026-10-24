import pytest

ZERO_RUN = "Ran 0 tests in 0.000s\n\nNO TESTS RAN\n"
OWN_MODULE = ("ERROR: test_cache (unittest.loader._FailedTest.test_cache)\n"
              "ImportError: Failed to import test module: test_cache\n"
              "ModuleNotFoundError: No module named 'tests.test_cache'\n\nRan 1 test in 0.000s\n\nFAILED (errors=1)\n")
PASSING = "tests/test_cache.py ...      [100%]\n\n3 passed in 0.01s\n"


@pytest.mark.parametrize("broken,superseded", [
    (ZERO_RUN, True),                                                     # the runner found nothing to run
    ("['site-packages']\nSTDERR:\n/usr/bin/python3: No module named pytest\n", True),  # no runner in that interpreter
    (OWN_MODULE, True),                                                   # it could not import its own test module
    (OWN_MODULE.replace("'tests.test_cache'", "'cache.lru'"), False),     # a code module is missing: real defect
    ("FAIL: test_a (t.T)\nAssertionError: 1 != 2\nRan 1 test in 0.001s\n\nFAILED (failures=1)\n", False),
])
def test_a_runner_that_ran_nothing_is_replaced_by_tests_that_ran(broken, superseded):
    from rune.agent.verification_state import VerificationState

    state = VerificationState()
    state.changed()
    state.observe_command("python3 -m unittest discover -s tests", False, broken, "/w")
    state.observe_command("python3 -m pytest -q tests", True, PASSING, "/w")
    assert state.passed is superseded


def test_a_narrower_passing_run_does_not_replace_a_wider_one_that_ran_nothing():
    from rune.agent.verification_state import VerificationState

    state = VerificationState()
    state.changed()
    state.observe_command("python3 -m unittest discover -s tests", False, ZERO_RUN, "/w")
    state.observe_command("python3 -m pytest -q tests/test_cache.py", True, PASSING, "/w")
    assert not state.passed


def test_a_run_that_ran_nothing_after_a_covering_pass_changes_nothing():
    from rune.agent.verification_state import VerificationState

    state = VerificationState()
    state.observe_command("python3 -m pytest -v tests/test_slug.py", True, PASSING, "/w")
    state.observe_command("python3 -m unittest -v tests.test_slug", True, ZERO_RUN, "/w")
    assert not state.pending
    # A real failure afterwards still counts.
    state.observe_command("python3 -m pytest -v tests/test_slug.py", False, "1 failed, 2 passed", "/w")
    assert state.pending
