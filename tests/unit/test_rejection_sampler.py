"""Tests for verifier-guided rejection sampling (best-of-K)."""

from __future__ import annotations

import os

import pytest

from rune.agent.rejection_sampler import (
    make_evidence_gate_verifier,
    make_verifier,
    sample_parallel,
    solve_with_rejection,
)


def _runner(values: list[str]):
    async def run_attempt(i: int) -> str:
        return values[i]

    return run_attempt


def _verify_equals(good: str):
    async def verify(candidate: str) -> bool:
        return candidate == good

    return verify


@pytest.mark.asyncio
async def test_selects_first_passing_and_stops() -> None:
    # Candidates: fail, fail, PASS, (4th never sampled due to early stop).
    run = _runner(["bad", "bad", "good", "good"])
    res = await solve_with_rejection(run, _verify_equals("good"), k=4)
    assert res.solved
    assert res.selected == "good"
    assert res.selected_index == 2
    assert len(res.attempts) == 3  # stopped at first pass


@pytest.mark.asyncio
async def test_no_pass_returns_unsolved() -> None:
    run = _runner(["bad", "bad", "bad"])
    res = await solve_with_rejection(run, _verify_equals("good"), k=3)
    assert not res.solved
    assert res.selected is None
    assert res.selected_index is None
    assert res.pass_count == 0
    assert len(res.attempts) == 3


@pytest.mark.asyncio
async def test_sample_all_counts_pass_rate() -> None:
    # stop_on_first_pass=False samples all k to measure the rate.
    run = _runner(["good", "bad", "good", "bad"])
    res = await solve_with_rejection(
        run, _verify_equals("good"), k=4, stop_on_first_pass=False
    )
    assert res.solved
    assert res.selected_index == 0  # first pass still recorded as selected
    assert res.pass_count == 2
    assert len(res.attempts) == 4


@pytest.mark.asyncio
async def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        await solve_with_rejection(_runner([]), _verify_equals("x"), k=0)


@pytest.mark.asyncio
async def test_parallel_samples_all_and_selects_lowest_index() -> None:
    run = _runner(["bad", "good", "bad", "good"])
    res = await sample_parallel(run, _verify_equals("good"), k=4)
    assert res.solved
    assert res.selected_index == 1  # lowest-index pass
    assert res.pass_count == 2
    assert len(res.attempts) == 4  # all sampled (no early stop)


@pytest.mark.asyncio
async def test_parallel_no_pass_unsolved() -> None:
    run = _runner(["bad", "bad"])
    res = await sample_parallel(run, _verify_equals("good"), k=2)
    assert not res.solved
    assert res.pass_count == 0


@pytest.mark.asyncio
async def test_evidence_gate_verifier_selects_only_pass(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return "echo check"

    async def fake_run(script: str, cwd: str):
        return ("pass" if cwd == "good" else "fail", "")

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    monkeypatch.setattr(eg, "run_evidence_check", fake_run)

    verify = await make_evidence_gate_verifier("task")
    assert await verify("good") is True
    assert await verify("bad") is False

    # skip is treated as not-selected (conservative). New verifier picks up the
    # patched run (the closure captures run_evidence_check at build time).
    async def fake_run_skip(script: str, cwd: str):
        return ("skip", "")

    monkeypatch.setattr(eg, "run_evidence_check", fake_run_skip)
    verify_skip = await make_evidence_gate_verifier("task")
    assert await verify_skip("good") is False


@pytest.mark.asyncio
async def test_evidence_gate_verifier_records_failure_evidence(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return "echo check"

    async def fake_run(script: str, cwd: str):
        # "bad" fails with mismatch evidence; "good" passes.
        if cwd == "good":
            return ("pass", "")
        return ("fail", f"mismatch at {cwd}")

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    monkeypatch.setattr(eg, "run_evidence_check", fake_run)

    verify = await make_evidence_gate_verifier("task")
    assert await verify("bad") is False
    assert await verify("good") is True
    # the failed candidate's evidence is captured (for best-of failure learning),
    # the passing one leaves no evidence
    assert verify.evidence_by_cwd == {"bad": "mismatch at bad"}


@pytest.mark.asyncio
async def test_evidence_gate_verifier_no_check_never_selects(monkeypatch) -> None:
    import rune.agent.evidence_gate as eg

    async def fake_extract(instruction: str):
        return None  # NO_CHECK

    monkeypatch.setattr(eg, "extract_success_check", fake_extract)
    verify = await make_evidence_gate_verifier("task")
    assert await verify("anything") is False


class TestMakeVerifier:
    """Execution-first verifier: repo tests select; Evidence Gate is the fallback."""

    @pytest.mark.asyncio
    async def test_prefers_tests_pass(self, monkeypatch) -> None:
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", _no_eg)
        import rune.agent.auto_verify as av
        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("pass", "") if cwd == "good" else ("fail", "1 failed")

        monkeypatch.setattr(av, "run_verify", fake_run)
        verify = await make_verifier("task")
        assert await verify("good") is True
        assert await verify("bad") is False
        assert verify.evidence_by_cwd == {"bad": "1 failed"}  # test output kept

    @pytest.mark.asyncio
    async def test_falls_back_to_eg_when_no_tests(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

        async def fake_eg(instruction):
            async def v(cwd):
                return cwd == "eg_good"
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("eg_good") is True
        assert await verify("eg_bad") is False

    @pytest.mark.asyncio
    async def test_skip_falls_through_to_eg(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("skip", "")  # could not run tests

        monkeypatch.setattr(av, "run_verify", fake_run)

        async def fake_eg(instruction):
            async def v(cwd):
                return True  # EG accepts when tests are inconclusive
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("anywhere") is True

    @pytest.mark.asyncio
    async def test_has_check_true_when_seed_has_tests(self, monkeypatch) -> None:
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", _no_eg)
        monkeypatch.setattr(av, "detect_test_command", lambda cwd: ["pytest"])
        verify = await make_verifier("task", seed_cwd="/repo")
        assert verify.has_check is True  # tests in the seed = a check exists

    @pytest.mark.asyncio
    async def test_records_which_method_decided(self, monkeypatch) -> None:
        # The UX line "picked #i (passed `pytest -q`)" depends on the verifier
        # recording WHAT decided each candidate: the test command when tests
        # ran (pass or fail), the Evidence Gate when tests were unavailable.
        import rune.agent.auto_verify as av
        import rune.agent.rejection_sampler as rs

        def fake_detect(cwd):
            return ["pytest", "-q"] if cwd in ("good", "bad") else None

        async def fake_run(cmd, cwd, timeout=60.0):
            return ("pass", "") if cwd == "good" else ("fail", "1 failed")

        monkeypatch.setattr(av, "detect_test_command", fake_detect)
        monkeypatch.setattr(av, "run_verify", fake_run)

        async def fake_eg(instruction):
            async def v(cwd):
                return True
            v.has_check = True
            v.evidence_by_cwd = {}
            return v

        monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)
        verify = await make_verifier("task")
        assert await verify("good") is True
        assert await verify("bad") is False
        assert await verify("no_tests") is True
        assert verify.method_by_cwd == {
            "good": "`pytest -q`",
            "bad": "`pytest -q`",
            "no_tests": "Evidence Gate",
        }


async def _no_eg(instruction):
    """An Evidence Gate that has no check and never selects."""
    async def v(cwd):
        return False
    v.has_check = False
    v.evidence_by_cwd = {}
    return v


# --- targeted-test verification + eg_disabled -------------------------------


def _seed_and_candidate(tmp_path, *, edit=True, agent_test=False):
    """A seed tree (pkg/mod.py + pkg/tests/test_mod.py) and a candidate copy
    whose pkg/mod.py was edited (content + mtime differ)."""
    import os
    import shutil
    import time

    seed = tmp_path / "seed"
    (seed / "pkg" / "tests").mkdir(parents=True)
    (seed / "pkg" / "mod.py").write_text("x = 1\n")
    (seed / "pkg" / "tests" / "test_mod.py").write_text("def test_x():\n    pass\n")
    cand = tmp_path / "cand"
    shutil.copytree(seed, cand, copy_function=shutil.copy2)
    if edit:
        p = cand / "pkg" / "mod.py"
        p.write_text("x = 2\n")
        os.utime(p, (time.time() + 5, time.time() + 5))
    if agent_test:
        (cand / "pkg" / "test_agent_written.py").write_text("def test_a(): pass\n")
    return str(seed), str(cand)


def test_targeted_test_files_maps_changed_source(tmp_path):
    from rune.agent.rejection_sampler import _targeted_test_files

    seed, cand = _seed_and_candidate(tmp_path)
    assert _targeted_test_files(cand, seed) == [
        os.path.join("pkg", "tests", "test_mod.py")
    ]


def test_targeted_test_files_ignores_agent_written_tests(tmp_path):
    from rune.agent.rejection_sampler import _targeted_test_files

    # Candidate adds its own test file; it does not exist in the seed, so it
    # must never be used as evidence.
    seed, cand = _seed_and_candidate(tmp_path, edit=False, agent_test=True)
    assert _targeted_test_files(cand, seed) == []


def test_restore_canonical_tests_overwrites_tampered_copy(tmp_path):
    from rune.agent.rejection_sampler import _restore_canonical_tests

    seed, cand = _seed_and_candidate(tmp_path)
    tampered = os.path.join(cand, "pkg", "tests", "test_mod.py")
    open(tampered, "w").write("def test_x():\n    assert True  # gutted\n")
    _restore_canonical_tests(cand, seed, [os.path.join("pkg", "tests", "test_mod.py")])
    assert open(tampered).read() == "def test_x():\n    pass\n"


@pytest.mark.asyncio
async def test_targeted_tests_reject_and_pass(monkeypatch, tmp_path):
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

    calls: list[list[str]] = []
    verdict = {"state": "fail", "evidence": "1 failed, 2 passed in 0.1s"}

    async def fake_run(cmd, cwd, timeout=60.0):
        calls.append(cmd)
        return verdict["state"], verdict["evidence"]

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return True  # EG would vacuously pass — must not be reached

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    # Real failed tests reject the candidate (EG never consulted).
    assert await verify(cand) is False
    assert verify.evidence_by_cwd[cand] == "1 failed, 2 passed in 0.1s"
    assert any("test_mod.py" in " ".join(c) for c in calls)

    # A passing run with real assertions verifies.
    verdict.update(state="pass", evidence="3 passed in 0.2s")
    verify2 = await make_verifier("task", seed_cwd=seed)
    assert await verify2(cand) is True
    assert "targeted tests" in verify2.method_by_cwd[cand]


@pytest.mark.asyncio
async def test_targeted_collection_error_is_inconclusive(monkeypatch, tmp_path):
    # pytest exiting non-zero WITHOUT "N failed" (import/collection error, e.g.
    # interpreter mismatch) must not count as a rejection; flow falls through
    # to the Evidence Gate.
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

    async def fake_run(cmd, cwd, timeout=60.0):
        return "fail", "ImportError: No module named 'distutils'"

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return True

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    assert await verify(cand) is True  # fell through to EG
    assert verify.method_by_cwd[cand] == "Evidence Gate"


@pytest.mark.asyncio
async def test_eg_disabled_blocks_vacuous_eg_pass(monkeypatch, tmp_path):
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

    async def fake_eg(instruction):
        async def v(cwd):
            return True  # vacuous pass

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task")
    assert await verify(str(tmp_path)) is True  # EG allowed by default
    verify.eg_disabled = True
    assert await verify(str(tmp_path)) is False  # vacuous pass suppressed


def test_project_python_prefers_repo_venv(tmp_path):
    import sys

    from rune.agent.rejection_sampler import _project_python
    # No venv → RUNE's own interpreter.
    assert _project_python(str(tmp_path)) == sys.executable
    # Project venv present → its python wins.
    vbin = tmp_path / ".venv" / "bin"
    vbin.mkdir(parents=True)
    py = vbin / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    assert _project_python(str(tmp_path)) == str(py)


@pytest.mark.asyncio
async def test_targeted_tests_use_project_interpreter(monkeypatch, tmp_path):
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    vbin = tmp_path / "seed" / ".venv" / "bin"
    vbin.mkdir(parents=True)
    proj_py = vbin / "python"
    proj_py.write_text("#!/bin/sh\n")
    proj_py.chmod(0o755)

    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)
    calls: list[list[str]] = []

    async def fake_run(cmd, cwd, timeout=60.0):
        calls.append(cmd)
        return "pass", "3 passed in 0.1s"

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return False

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    assert await verify(cand) is True
    cmd = calls[0]
    assert str(proj_py) in cmd          # project venv interpreter used
    assert any(c.startswith("PYTHONPATH=") and cand in c for c in cmd)


# --- layered test mapper -----------------------------------------------------


def _mk(root, rel, text=""):
    import os
    p = root / rel
    os.makedirs(p.parent, exist_ok=True)
    p.write_text(text)


def test_mapper_sphinx_path_join_layout(tmp_path):
    # sphinx/ext/autodoc/importer.py must map to tests/test_ext_autodoc.py
    from rune.agent.rejection_sampler import _targeted_test_files

    seed = tmp_path / "seed"
    _mk(seed, "sphinx/ext/autodoc/importer.py", "x=1\n")
    _mk(seed, "tests/test_ext_autodoc.py", "def test_a(): pass\n")
    cand = tmp_path / "cand"
    import shutil
    shutil.copytree(seed, cand, copy_function=shutil.copy2)
    p = cand / "sphinx/ext/autodoc/importer.py"
    p.write_text("x=2\n")
    import os
    import time
    os.utime(p, (time.time() + 5, time.time() + 5))

    assert "tests/test_ext_autodoc.py" in _targeted_test_files(str(cand), str(seed))


def test_mapper_import_grep_finds_unrelated_name(tmp_path):
    # Test file with an unconventional name that IMPORTS the changed module.
    from rune.agent.rejection_sampler import _targeted_test_files

    seed = tmp_path / "seed"
    _mk(seed, "pkg/core/engine.py", "x=1\n")
    _mk(seed, "tests/test_smoke_suite.py",
        "from pkg.core.engine import run\ndef test_r(): pass\n")
    cand = tmp_path / "cand"
    import shutil
    shutil.copytree(seed, cand, copy_function=shutil.copy2)
    p = cand / "pkg/core/engine.py"
    p.write_text("x=2\n")
    import os
    import time
    os.utime(p, (time.time() + 5, time.time() + 5))

    out = _targeted_test_files(str(cand), str(seed))
    assert out and out[0] == "tests/test_smoke_suite.py"  # L1 ranks first


def test_mapper_django_dir_token_layout(tmp_path):
    # django/db/models/query.py → tests/queries/tests.py via token match.
    from rune.agent.rejection_sampler import _targeted_test_files

    seed = tmp_path / "seed"
    _mk(seed, "django/db/models/query.py", "x=1\n")
    _mk(seed, "tests/queries/tests.py", "def test_q(): pass\n")
    _mk(seed, "tests/migrations/tests.py", "def test_m(): pass\n")
    cand = tmp_path / "cand"
    import shutil
    shutil.copytree(seed, cand, copy_function=shutil.copy2)
    p = cand / "django/db/models/query.py"
    p.write_text("x=2\n")
    import os
    import time
    os.utime(p, (time.time() + 5, time.time() + 5))

    out = _targeted_test_files(str(cand), str(seed))
    assert "tests/queries/tests.py" in out
    assert "tests/migrations/tests.py" not in out


@pytest.mark.asyncio
async def test_suite_collection_error_falls_through_to_targeted(monkeypatch, tmp_path):
    # Top-level tests/ triggers the full-suite path; a collection error there
    # (no "N failed") must fall through to targeted tests, not reject.
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    monkeypatch.setattr(
        av, "detect_test_command", lambda cwd: ["python", "-m", "pytest", "-q"]
    )

    calls: list[list[str]] = []

    async def fake_run(cmd, cwd, timeout=60.0):
        calls.append(cmd)
        if len(calls) == 1:  # full-suite attempt: import explosion, no "failed"
            return "fail", "ImportError: cannot import name 'x'\nexit 2"
        return "pass", "4 passed in 0.2s"  # targeted run

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return False

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    assert await verify(cand) is True  # verified via targeted, not rejected
    assert "targeted tests" in verify.method_by_cwd[cand]


@pytest.mark.asyncio
async def test_suite_verdict_memoized_across_candidates(monkeypatch, tmp_path):
    # An inconclusive full-suite run (timeout/collection error) must not be
    # re-paid for every sibling candidate (each repeat costs the full timeout).
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    detect_calls = {"n": 0}

    def fake_detect(cwd):
        detect_calls["n"] += 1
        return ["python", "-m", "pytest", "-q"]

    monkeypatch.setattr(av, "detect_test_command", fake_detect)

    suite_runs = {"n": 0}

    async def fake_run(cmd, cwd, timeout=60.0):
        if "-q" in cmd and not any("test_mod" in c for c in cmd):
            suite_runs["n"] += 1
            return "skip", ""  # timeout — inconclusive
        return "fail", "1 failed in 0.1s"  # targeted rejects

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return False

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    await verify(cand)
    await verify(cand)
    await verify(cand)
    assert suite_runs["n"] == 1  # paid once, memoized for siblings


@pytest.mark.asyncio
async def test_repro_script_verifies_fixing_candidate(monkeypatch, tmp_path):
    # A baseline-failing repro attached to the verifier: candidate that makes
    # it pass (and breaks no targeted tests) is VERIFIED; one that doesn't is
    # rejected with the repro output as evidence.
    import rune.agent.auto_verify as av
    import rune.agent.rejection_sampler as rs

    seed, cand = _seed_and_candidate(tmp_path)
    monkeypatch.setattr(av, "detect_test_command", lambda cwd: None)

    async def fake_run(cmd, cwd, timeout=60.0):
        return "pass", "2 passed in 0.1s"  # targeted regressions fine

    monkeypatch.setattr(av, "run_verify", fake_run)

    async def fake_eg(instruction):
        async def v(cwd):
            return False

        v.has_check = True
        v.evidence_by_cwd = {}
        return v

    monkeypatch.setattr(rs, "make_evidence_gate_verifier", fake_eg)

    verify = await make_verifier("task", seed_cwd=seed)
    # Repro checks the candidate's mod.py content.
    verify.repro_script = (
        "import sys, os\nsys.path.insert(0, os.getcwd())\n"
        "src = open('pkg/mod.py').read()\nassert 'x = 2' in src\n"
    )
    assert await verify(cand) is True  # cand has x = 2 → repro passes
    assert "reproduction script" in verify.method_by_cwd[cand]

    # Now a candidate that does NOT fix (seed copy has x = 1):
    import shutil as _sh
    unfixed = tmp_path / "unfixed"
    _sh.copytree(seed, unfixed, copy_function=_sh.copy2)
    assert await verify(str(unfixed)) is False
    assert "AssertionError" in verify.evidence_by_cwd[str(unfixed)]
