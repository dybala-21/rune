"""A run that removed user files the request never named is not a confirmed success."""

from rune.agent.action_receipts import ReceiptLedger
from rune.agent.run_outcome import merge_requirement_checks, run_outcome
from rune.types import CapabilityResult


def _shell(ledger, root, removed):
    ledger.record("bash_execute", {"command": "rm -rf ...", "cwd": str(root)},
                  CapabilityResult(success=True, metadata={"shell_changes": {"created": [], "modified": [], "removed": removed}}),
                  call_id="c1")


def test_a_vague_cleanup_that_removed_user_files_needs_review(tmp_path):
    ledger = ReceiptLedger()
    _shell(ledger, tmp_path, ["notebooks/analysis.ipynb", "data/backup.sql", "build/a.o", ".DS_Store"])

    check, note = ledger.removal_review("작업 디렉터리가 너무 지저분해. 정리 좀 해줘.", "ko")

    targets = [item["target"] for item in check["requirements"]]
    assert targets == [str(tmp_path / "data/backup.sql"), str(tmp_path / "notebooks/analysis.ipynb")]
    assert check["status"] == "unverified" and "2개" in note
    outcome = run_outcome({"reason": "completed", "requirement_acceptance": merge_requirement_checks(None, check)})
    assert outcome.completion == "completed" and not outcome.success and outcome.status == "unverified"


def test_files_the_request_named_by_folder_name_or_extension_pass(tmp_path):
    ledger = ReceiptLedger()
    _shell(ledger, tmp_path, ["logs/run_1.log", "build/tmp_a.o", "old/report.txt", "drafts/x.bak"])
    request = "Delete the .o files in build/, the old logs, report.txt and every .bak file."
    assert ledger.removal_review(request) is None


def test_regenerable_output_alone_never_needs_review(tmp_path):
    ledger = ReceiptLedger()
    _shell(ledger, tmp_path, ["__pycache__/m.cpython-313.pyc", "build/x.o", ".pytest_cache/v/cache"])
    assert ledger.removal_review("clean up") is None


def test_file_delete_is_reviewed_with_its_trash_copy(tmp_path):
    ledger = ReceiptLedger()
    target = tmp_path / "notes.md"
    ledger.record("file_delete", {"path": str(target)},
                  CapabilityResult(success=True, output="Moved to trash",
                                   metadata={"path": str(target), "trashed": str(tmp_path / ".rune-trash/notes.md")}),
                  call_id="c2")
    check, note = ledger.removal_review("tidy up the folder")
    assert check["requirements"][0]["target"] == str(target)
    assert ".rune-trash/notes.md" in note


def test_a_failed_delete_is_not_a_removal(tmp_path):
    ledger = ReceiptLedger()
    ledger.record("file_delete", {"path": str(tmp_path / "keep.md")},
                  CapabilityResult(success=False, error="denied"), call_id="c3")
    assert ledger.removal_review("tidy up") is None


def test_the_workspace_location_itself_never_counts_as_naming(tmp_path):
    """The workspace folder's own name does not count as naming the files removed inside it."""
    root = tmp_path / "work"
    ledger = ReceiptLedger()
    _shell(ledger, root, ["notes/manual.md"])
    assert ledger.removal_review("clean up my work folder") is not None
