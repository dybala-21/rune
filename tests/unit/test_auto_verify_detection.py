

def test_passed_test_count_parsing():
    from rune.agent.auto_verify import passed_test_count
    assert passed_test_count("3 passed in 0.01s") == 3
    assert passed_test_count("1 passed, 2 warnings in 0.1s") == 1
    assert passed_test_count("47 passed in 2.3s") == 47
    assert passed_test_count("no tests ran in 0.0s") is None
    assert passed_test_count("") is None


def test_assertions_ran_detects_real_runs():
    from rune.agent.auto_verify import assertions_ran
    assert assertions_ran("3 passed in 0.01s") is True
    assert assertions_ran("test result: ok. 7 passed; 0 failed") is True
    assert assertions_ran("Tests: 5 passed, 5 total") is True
    assert assertions_ran("Ran 4 tests in 0.001s") is True
    assert assertions_ran("ok  mypkg 0.002s") is True


def test_assertions_ran_detects_vacuous_runs():
    """Exit 0 with nothing asserted must not read as verification."""
    from rune.agent.auto_verify import assertions_ran
    assert assertions_ran("no tests ran in 0.0s") is False
    assert assertions_ran("test result: ok. 0 passed; 0 failed; 0 ignored") is False
    assert assertions_ran("collected 0 items") is False
    assert assertions_ran("Ran 0 tests in 0.000s") is False
    assert assertions_ran("Tests: 0 total") is False


def test_assertions_ran_unknown_stays_unknown():
    """Unparseable summaries must return None so callers don't over-block."""
    from rune.agent.auto_verify import assertions_ran
    assert assertions_ran("Build succeeded") is None
    assert assertions_ran("All checks complete") is None
    assert assertions_ran("") is None
    assert assertions_ran("   ") is None


def test_generated_test_refuses_to_clobber_existing_file(tmp_path):
    """The test file is deleted after the run, so overwriting one destroys it."""
    import asyncio

    from rune.agent.generated_test import Framework, run_generated_test

    fw = Framework(
        name="pytest", language="python",
        test_path="tests/test_rune_verify.py",
        command=["true"],
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    mine = tests_dir / "test_rune_verify.py"
    mine.write_text("USER'S OWN TEST")

    state, _ = asyncio.run(run_generated_test("assert True", fw, str(tmp_path)))

    assert state == "skip"
    assert mine.read_text() == "USER'S OWN TEST"


def test_discriminates_leaves_the_baseline_tree_untouched(tmp_path):
    """The baseline can be the user's project; probing it must not write there."""
    import asyncio

    from rune.agent.generated_test import Framework, discriminates

    fw = Framework(
        name="pytest", language="python",
        test_path="tests/test_rune_verify.py",
        command=["false"],  # non-zero => "fail" => discriminating
    )
    (tmp_path / "keep.py").write_text("x")
    before = sorted(p.name for p in tmp_path.iterdir())

    assert asyncio.run(discriminates("assert True", fw, str(tmp_path))) is True
    assert sorted(p.name for p in tmp_path.iterdir()) == before
