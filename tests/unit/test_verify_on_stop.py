"""Verify-on-stop: one bounded reminder when the model finishes with a prose
answer after editing code without running any test since the last edit.

Without this gate a model can edit, skip the tests, and declare the work
done — the reminder buys one test-and-repair round before the run ends.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rune.agent.litellm_adapter as la
from rune.agent.litellm_adapter import StreamResult


def _delta_chunk(*, content=None, tool_calls=None, finish_reason=None):
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                index=tc["index"],
                id=tc.get("id", f"tc{tc['index']}"),
                function=SimpleNamespace(
                    name=tc.get("name"), arguments=tc.get("arguments")
                ),
            )
            for tc in tool_calls
        ]
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tc_objs),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=None)


async def _astream(chunks):
    for c in chunks:
        yield c


def _edit_turn():
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "file_edit",
            "arguments": '{"path": "src/mod.py", "search": "a", "replace": "b"}',
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


def _bash_turn(command: str):
    import json as _json
    return [
        _delta_chunk(tool_calls=[{
            "index": 0, "name": "bash_execute",
            "arguments": _json.dumps({"command": command}),
        }]),
        _delta_chunk(finish_reason="tool_calls"),
    ]


def _final_turn(text="done, the fix is applied"):
    return [
        _delta_chunk(content=text),
        _delta_chunk(finish_reason="stop"),
    ]


def _make_result():
    async def edit_tool(**_):
        return "edited"

    async def bash_tool(**_):
        return "2 passed in 0.1s"

    return StreamResult(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "fix the bug"}],
        tool_schemas=[
            {"function": {"name": "file_edit"}},
            {"function": {"name": "bash_execute"}},
        ],
        tool_lookup={"file_edit": edit_tool, "bash_execute": bash_tool},
        max_tokens=8192,
        temperature=0.0,
        request_tokens_limit=200000,
        response_tokens_limit=8192,
        max_tool_rounds=20,
    )


def _fake_acompletion(streams):
    it = iter(streams)

    async def fake(**kwargs):
        return _astream(next(it))

    return fake


def _vos_messages(result):
    return [
        m for m in result.all_messages()
        if m.get("role") == "user"
        and (m.get("content") == la._VERIFY_ON_STOP_MSG
             or "stopped without running any test" in str(m.get("content")))
    ]


@pytest.mark.asyncio
async def test_nudges_once_when_finishing_untested_edit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    # edit → tries to finish → nudged → runs tests → finishes.
    streams = [
        _edit_turn(),
        _final_turn(),
        _bash_turn("python -m pytest tests/test_mod.py -q"),
        _final_turn("all tests passed"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_vos_messages(result)) == 1
    # final answer survived after the verification round
    assert "passed" in result._collected_text


@pytest.mark.asyncio
async def test_no_nudge_when_tests_ran_after_edit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    streams = [
        _edit_turn(),
        _bash_turn("pytest -q"),
        _final_turn(),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _vos_messages(result) == []


@pytest.mark.asyncio
async def test_nudge_is_bounded_to_one(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
    result = _make_result()
    # Model ignores the nudge, edits again, and finishes untested — the
    # second finish must NOT be blocked (bounded at one reminder).
    streams = [
        _edit_turn(),
        _final_turn(),
        _edit_turn(),
        _final_turn("shipping anyway"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert len(_vos_messages(result)) == 1


@pytest.mark.asyncio
async def test_env_opt_out(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_VERIFY_ON_STOP", "0")
    result = _make_result()
    streams = [_edit_turn(), _final_turn()]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))

    async for _ in result.stream_text():
        pass

    assert _vos_messages(result) == []


def test_is_test_command_matches_runners():
    assert la._is_test_command("python -m pytest tests/ -q")
    assert la._is_test_command("cd repo && npm test")
    assert la._is_test_command("cargo test --lib")
    assert la._is_test_command("python tests/runtests.py --parallel 1 x")
    assert not la._is_test_command("grep -r pytest docs/")
    assert not la._is_test_command("pip install pytest")


class TestExecutedStopCheck:
    """The reminder now runs the check instead of asking for it.

    Words can be ignored — several measured nudges changed nothing — and
    they cost two rounds when obeyed. The executed form hands the model the
    verdict. What must never happen: the whole suite running on a large
    repository, where the timeout would cost more than the rounds saved.
    """

    def test_targets_are_matched_by_edited_stem(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_billing.py").write_text("def test_a(): pass\n")
        (tmp_path / "tests" / "test_other.py").write_text("def test_b(): pass\n")
        hits = la._stop_check_targets(str(tmp_path), {"src/billing.py"})
        assert hits and "test_billing.py" in hits[0]
        assert all("test_other" not in h for h in hits)

    def test_no_matching_test_file_means_no_run(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_other.py").write_text("def test_b(): pass\n")
        assert la._stop_check_targets(str(tmp_path), {"src/billing.py"}) == []

    def test_an_edited_test_file_does_not_match_itself(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_billing.py").write_text("def test_a(): pass\n")
        assert la._stop_check_targets(str(tmp_path), {"tests/test_billing.py"}) == []

    @pytest.mark.asyncio
    async def test_non_pytest_runner_falls_back_to_words(self, tmp_path, monkeypatch):
        import rune.agent.auto_verify as av
        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["npm", "test"])
        assert await la._run_stop_check(str(tmp_path), {"src/app.js"}) is None

    @pytest.mark.asyncio
    async def test_no_targets_never_runs_the_whole_suite(self, tmp_path, monkeypatch):
        import rune.agent.auto_verify as av
        monkeypatch.setattr(av, "detect_test_command",
                            lambda cwd: ["python", "-m", "pytest", "-q"])
        # No test files at all: executing here would mean the full suite.
        assert await la._run_stop_check(str(tmp_path), {"src/mod.py"}) is None

    @pytest.mark.asyncio
    async def test_executed_check_is_injected_with_its_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
        monkeypatch.delenv("RUNE_VOS_EXEC", raising=False)

        async def fake_check(cwd, edited, allowed=None):
            assert "src/mod.py" in edited
            return ("python -m pytest tests/test_mod.py -q",
                    "1 failed: expected 3 got 2", 1)
        monkeypatch.setattr(la, "_run_stop_check", fake_check)

        result = _make_result()
        streams = [_edit_turn(), _final_turn(),
                   _final_turn("fixed and done")]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass

        vos = _vos_messages(result)
        assert len(vos) == 1
        body = str(vos[0]["content"])
        assert "test_mod.py" in body           # the model sees what ran
        assert "expected 3 got 2" in body      # and what it said


class TestMechanicalCheckVerdict:
    """The last test execution's verdict outlives the model's summary.

    Measured false-done: the transcript showed "4 failed", the model closed
    with a confident summary anyway, and the process exited 0. The verdict
    channel makes that impossible to do quietly: whoever ran the tests —
    the harness at a stop, or the model through bash — the last mechanical
    outcome is what the end of the run answers to.
    """

    @pytest.mark.asyncio
    async def test_a_failing_test_run_is_remembered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
        la.consume_mech_check()   # clear whatever an earlier test left

        result = _make_result()

        async def failing_bash(**_):
            return "Error: exit 1\n2 failed, 1 passed"
        result._tool_lookup["bash_execute"] = failing_bash

        # The failing run leaves the reminder armed (correct), so one more
        # turn is consumed by the nudge before the final answer.
        streams = [_edit_turn(), _bash_turn("python -m pytest tests/ -q"),
                   _final_turn("all good, shipping"),
                   _final_turn("all good, shipping")]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass

        assert la.consume_mech_check() == "fail"

    @pytest.mark.asyncio
    async def test_a_passing_rerun_clears_the_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
        la.consume_mech_check()

        result = _make_result()
        outputs = iter(["Error: exit 1\n1 failed", "3 passed in 0.2s"])

        async def bash(**_):
            return next(outputs)
        result._tool_lookup["bash_execute"] = bash

        streams = [_edit_turn(), _bash_turn("pytest -q"),
                   _bash_turn("pytest -q"), _final_turn()]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass

        assert la.consume_mech_check() == "pass"

    @pytest.mark.asyncio
    async def test_non_test_bash_says_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
        la.consume_mech_check()

        result = _make_result()

        async def bash(**_):
            return "Error: rm: cannot remove"
        result._tool_lookup["bash_execute"] = bash

        streams = [_edit_turn(), _bash_turn("ls -la"), _final_turn(),
                   _final_turn()]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass

        assert la.consume_mech_check() == ""

    def test_consume_clears(self):
        la._MECH_CHECK.set("fail")
        assert la.consume_mech_check() == "fail"
        assert la.consume_mech_check() == ""


class TestFinalHarnessCheck:
    """After edits, the harness gets the last word, not the model.

    Measured: the stop check reported a failure, the model then ran a
    narrower command of its own that passed, and that pass overwrote the
    verdict — exit 0 with the real tests still failing. The final check
    re-runs the covering tests that existed BEFORE the editing began; a test
    file authored during the run answers for nothing.
    """

    @pytest.mark.asyncio
    async def test_final_check_overrides_a_model_run_pass(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VERIFY_ON_STOP", raising=False)
        monkeypatch.delenv("RUNE_VOS_EXEC", raising=False)
        la.consume_mech_check()

        calls = []

        async def harness_check(cwd, edited, allowed=None):
            calls.append(allowed)
            return ("pytest tests/test_mod.py", "1 failed", 1)   # truth: FAIL
        monkeypatch.setattr(la, "_run_stop_check", harness_check)

        result = _make_result()
        # model runs its own narrower test and it "passes"
        streams = [_edit_turn(), _bash_turn("pytest my_own_check.py -q"),
                   _final_turn("all green, done")]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass

        assert la.consume_mech_check() == "fail"   # harness verdict wins
        assert calls, "final harness check never ran"

    @pytest.mark.asyncio
    async def test_no_edits_means_no_final_check(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VOS_EXEC", raising=False)
        la.consume_mech_check()
        calls = []

        async def harness_check(cwd, edited, allowed=None):
            calls.append(1)
            return ("x", "y", 0)
        monkeypatch.setattr(la, "_run_stop_check", harness_check)

        result = _make_result()
        streams = [_final_turn("nothing to change")]
        monkeypatch.setattr(la.litellm, "acompletion", _fake_acompletion(streams))
        async for _ in result.stream_text():
            pass
        assert calls == []

