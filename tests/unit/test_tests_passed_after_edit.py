"""The 'tests passing' rung: a real post-edit test pass, and nothing weaker."""

from types import SimpleNamespace

from rune.api.server import build_trust_payload


def _trace(**kw):
    base = {"reason": "completed", "evidence_gate": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_payload_carries_the_three_states():
    assert build_trust_payload(
        _trace(tests_passed_after_edit=True)
    )["testsPassedAfterEdit"] is True
    assert build_trust_payload(
        _trace(tests_passed_after_edit=False)
    )["testsPassedAfterEdit"] is False
    # Nothing was edited: the question does not apply, and a null must not be
    # rendered as either a pass or a failure.
    assert build_trust_payload(
        _trace(tests_passed_after_edit=None)
    )["testsPassedAfterEdit"] is None
    # Older traces without the field behave like "not applicable".
    assert build_trust_payload(_trace())["testsPassedAfterEdit"] is None


def test_signal_requires_an_asserting_test_after_the_edit():
    """Mirrors the loop's rule: a passing `ls` is not a passing test suite."""

    def resolve(last_code_write: int, last_test_pass: int) -> bool | None:
        # The finalize rule in NativeAgentLoop.run, isolated.
        if last_code_write <= 0:
            return None
        return last_test_pass > last_code_write

    # Edited, then a real test passed → the rung is earned.
    assert resolve(3, 5) is True
    # Edited, then only non-test commands ran (no test-pass step recorded).
    assert resolve(3, 0) is False
    # Test passed, then the code was edited again — the green is stale.
    assert resolve(7, 5) is False
    # Nothing was edited: vacuous, so not claimable either way.
    assert resolve(0, 4) is None
    assert resolve(0, 0) is None


def test_loop_records_the_pass_step_only_for_asserting_runs():
    """The recording site must sit behind the assertions_ran guard."""
    import inspect

    from rune.agent import loop as loop_mod

    src = inspect.getsource(loop_mod)
    marker = "self._last_test_pass_step = self._tool_call_seq"
    assert marker in src
    before = src.split(marker)[0]
    # The nearest preceding condition is the assertion check, so an empty
    # suite exiting 0 cannot register as a passing test.
    assert "assertions_ran(result.output or \"\") is not False" in before.split(
        "is_verification_command"
    )[-1]
