"""Completion verdicts on harsh coding requests, graded by gbench2's hidden checkers.

Fails only on a false done: RUNE reports success for work the checker rejects.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.test_live_workflows import workflow as workflow

GBENCH = Path("~/workspace/swe-mini-work/gbench2").expanduser()
CELLS = ["H1_already_fixed", "H2_contradicting_test", "H3_wrong_premise",
         "H4_regression_trap", "H5_destructive_cleanup", "H6_shell_cleanup"]

pytestmark = pytest.mark.skipif(not GBENCH.is_dir(), reason="gbench2 fixtures are not available")


@pytest.mark.parametrize("workflow", [{"max_iterations": 40, "timeout_seconds": 600}], indirect=True)
@pytest.mark.parametrize("cell", CELLS)
def test_harsh_coding_request_never_ends_in_a_false_done(workflow, cell, request):
    source = GBENCH / cell
    shutil.copytree(source / "fixture", workflow.workspace, dirs_exist_ok=True, symlinks=True)
    task = (source / "task.txt").read_text(encoding="utf-8").strip()

    run = workflow.submit(task, approve_all=True, wait_seconds=660)

    answer = run.get("answer") or run.get("text") or ""
    transcript = workflow.workspace.parent / f"{cell}.answer.txt"
    transcript.write_text(answer, encoding="utf-8")
    grader = subprocess.run([sys.executable, str(source / "check.py"), str(workflow.workspace), "0", str(transcript)],
                            capture_output=True, text=True, timeout=600)
    correct = grader.returncode == 0
    trust = run.get("trust") or {}
    claimed = run.get("success") is True
    verdict = {"cell": cell, "grader_pass": correct, "rune_success": claimed, "verified": trust.get("verified"),
               "reason": trust.get("reason"), "verification": trust.get("verificationStatus"),
               "grader": (grader.stdout or grader.stderr).strip()[:300], "answer_tail": answer[-400:],
               "tools": [f"{call.get('toolName')} {json.dumps(call.get('args') or {}, ensure_ascii=False)[:600]}"
                         f" => {str(call.get('result') or '')[:300]!r}" for call in run.get("toolCalls") or []],
               "approved": [str(item.get("command") or item)[:160] for item in workflow.runs[-1]["approved_fixture_edits"]]}
    print("HARSH_VERDICT " + json.dumps(verdict, ensure_ascii=False))
    request.node.live_checks = {"task_outcome": correct, "no_false_completion": not (claimed and not correct)}
    request.node.live_passed = correct
    assert not (claimed and not correct), f"False done: RUNE reported success but the hidden checker failed: {verdict}"
