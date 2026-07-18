"""The verifier must never judge against agent-edited verification inputs.

Covers the measured tampering shapes (scripts/tamper_rate_bench.py): an
assertion weakened so wrong code passes, a test corrupted so correct code
fails, a fresh conftest.py that alters collection, and runner-config edits.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from rune.agent.validation_guard import (
    restoration_note,
    restore_tests,
    snapshot_tests,
)

ORIGINAL = "def test_twenty_percent():\n    assert tiered_discount(1000) == 800\n"
TAMPERED = "def test_twenty_percent():\n    assert tiered_discount(1000) == 900\n"


def _seed(tmp_path: Path) -> Path:
    f = tmp_path / "test_discount.py"
    f.write_text(ORIGINAL)
    (tmp_path / "discount.py").write_text("def tiered_discount(a):\n    return a\n")
    return f


def test_tampered_assertion_is_restored(tmp_path):
    f = _seed(tmp_path)
    snap = snapshot_tests(tmp_path)
    f.write_text(TAMPERED)

    report = restore_tests(snap)

    assert report.restored == ["test_discount.py"]
    assert report.tampering()
    assert f.read_text() == ORIGINAL


def test_deleted_test_file_is_restored(tmp_path):
    f = _seed(tmp_path)
    snap = snapshot_tests(tmp_path)
    f.unlink()

    assert restore_tests(snap).restored == ["test_discount.py"]
    assert f.read_text() == ORIGINAL


def test_untouched_files_and_created_tests_are_left_but_disclosed(tmp_path):
    _seed(tmp_path)
    snap = snapshot_tests(tmp_path)
    created = tmp_path / "test_new.py"
    created.write_text("def test_added(): pass\n")

    report = restore_tests(snap)

    assert report.restored == []
    assert report.created == ["test_new.py"]
    assert not report.tampering()  # creation alone can't unfail an existing test
    assert created.read_text() == "def test_added(): pass\n"


def test_new_conftest_is_quarantined(tmp_path):
    """A fresh conftest.py can skip/deselect the failing tests — the collection
    attack. It must not be present when the verification runs."""
    _seed(tmp_path)
    snap = snapshot_tests(tmp_path)
    evil = tmp_path / "conftest.py"
    evil.write_text("collect_ignore = ['test_discount.py']\n")

    report = restore_tests(snap)

    assert report.quarantined == ["conftest.py"]
    assert report.tampering()
    assert not evil.exists()
    quarantined = list(tmp_path.glob("conftest.py.rune-quarantined*"))
    assert len(quarantined) == 1  # preserved for inspection, not collected


def test_preexisting_conftest_is_frozen_not_quarantined(tmp_path):
    _seed(tmp_path)
    conftest = tmp_path / "conftest.py"
    conftest.write_text("# fixtures\n")
    snap = snapshot_tests(tmp_path)
    conftest.write_text("collect_ignore = ['test_discount.py']\n")

    report = restore_tests(snap)

    assert report.restored == ["conftest.py"]
    assert report.quarantined == []
    assert conftest.read_text() == "# fixtures\n"


def test_runner_config_changes_are_disclosed_not_reverted(tmp_path):
    _seed(tmp_path)
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text("[tool.pytest.ini_options]\naddopts = ''\n")
    snap = snapshot_tests(tmp_path)
    sneaky = "[tool.pytest.ini_options]\naddopts = '--deselect test_discount.py'\n"
    cfg.write_text(sneaky)

    report = restore_tests(snap)

    assert report.configs_changed == ["pyproject.toml"]
    assert cfg.read_text() == sneaky  # legitimate edits are never clobbered
    assert "pyproject.toml" in restoration_note(report)


def test_non_test_files_are_not_snapshotted(tmp_path):
    _seed(tmp_path)
    snap = snapshot_tests(tmp_path)

    assert set(snap.files) == {"test_discount.py"}


def test_files_under_tests_dir_are_covered(tmp_path):
    sub = tmp_path / "tests"
    sub.mkdir()
    f = sub / "check_behavior.py"  # no test_* prefix; caught by directory rule
    f.write_text(ORIGINAL)
    snap = snapshot_tests(tmp_path)
    f.write_text(TAMPERED)

    assert restore_tests(snap).restored == [str(Path("tests") / "check_behavior.py")]
    assert f.read_text() == ORIGINAL


def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_PROTECT_TESTS", "0")
    f = _seed(tmp_path)
    assert snapshot_tests(tmp_path) is None
    f.write_text(TAMPERED)
    assert not restore_tests(None).anything()
    assert f.read_text() == TAMPERED


def test_walk_is_bounded(tmp_path, monkeypatch):
    """A mis-pinned huge workspace must not stall the run."""
    import rune.agent.validation_guard as vg

    for i in range(30):
        d = tmp_path / f"pkg{i:02d}"
        d.mkdir()
        (d / "test_x.py").write_text("def test_a(): pass\n")
    monkeypatch.setattr(vg, "_MAX_DIRS", 10)

    snap = snapshot_tests(tmp_path)

    assert snap is not None
    assert 0 < len(snap.files) < 30  # truncated, not hung


def test_restoration_note_mentions_files():
    from rune.agent.validation_guard import RestoreReport

    note = restoration_note(RestoreReport(restored=["test_discount.py"]))
    assert "test_discount.py" in note and "restored" in note
    assert restoration_note(RestoreReport()) == ""


def test_evidence_gate_discloses_restoration(tmp_path):
    from rune.agent.evidence_gate import EvidenceGate

    _seed(tmp_path)
    gate = EvidenceGate("fix the failing test", str(tmp_path))
    gate.last_tests_restored = ["test_discount.py"]
    gate._guard_note = restoration_note(
        __import__("rune.agent.validation_guard", fromlist=["RestoreReport"])
        .RestoreReport(restored=["test_discount.py"])
    )

    state, message = gate._record("fail", "blocked: check failed", "1 failed")

    assert state == "fail"
    assert "test_discount.py" in gate.last_evidence
    assert "test_discount.py" in (message or "")
    assert gate.summary()["tests_restored"] == ["test_discount.py"]


def test_goal_validate_restores_before_running(tmp_path):
    from rune.agent.goal_validate import make_validate_fn

    f = _seed(tmp_path)
    seen: list[str] = []

    async def fake_exec(cmd: str, cwd: str, timeout_s: float):
        seen.append(f.read_text())  # content AT command time
        return 0, "3 passed"

    validate = make_validate_fn(cwd=str(tmp_path), exec_fn=fake_exec, auto_root=False)
    f.write_text(TAMPERED)

    ok, detail = asyncio.run(validate(["pytest -q"]))

    assert ok
    assert seen == [ORIGINAL]  # the command saw the canonical test, not the tampered one
    assert "test_discount.py" in detail
    assert f.read_text() == ORIGINAL
