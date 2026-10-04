"""Calibrate requirement verdicts against configured models using synthetic output."""

import json
import time
from pathlib import Path

import pytest

from rune.agent.requirement_gate import check_adherence
from rune.agent.timing import capture_timing, timing_snapshot
from tests.e2e.test_live_workflows import workflow as workflow


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("requirements,output,expected", [
    (["Include exactly two bullet points", "Name Mina as the owner"],
     "- Owner: Mina\n- Deadline: Monday", "pass"),
    (["Include exactly two bullet points", "Name Mina as the owner"],
     "- Owner: Sam\n- Deadline: Monday", "fail"),
    (["Save the document as a single page and verify its rendered layout"],
     "I saved the document and checked that it fits on one page.\n"
     "[No file, rendered page, or inspection result was supplied.]", "skip"),
], ids=["supported", "mismatch", "unsupported_claim"])
async def test_requirement_verdict(live_model, monkeypatch, tmp_path, request, requirements, output, expected):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "state"))
    started = time.monotonic()
    with capture_timing() as timing:
        state, message = await check_adherence(requirements, output)
    report = {
        "provider": live_model[0], "model": live_model[1],
        "scope": "Real model requirement review; not a full application workflow",
        "expected": expected, "actual": state, "passed": state == expected,
        "seconds": round(time.monotonic() - started, 3), "detail": message,
        "timings": timing_snapshot(timing),
    }
    directory = request.config.getoption("--live-report-dir")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{live_model[0]}-requirements-{request.node.callspec.id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert state == expected, message or state


def test_report_review_reaches_web_completion(workflow, monkeypatch, request):
    monkeypatch.setenv("RUNE_REQUIREMENT_GATE", "1")
    run = workflow.submit(
        "Create brief.md with exactly two bullet points: owner Mina and deadline Monday. "
        "This is a short project brief. Verify the saved text, then give a short final reply.",
        editable_files=("brief.md",),
    )
    text = (workflow.workspace / "brief.md").read_text()
    assert "Mina" in text and "Monday" in text
    assert sum(line.lstrip().startswith(("- ", "* ")) for line in text.splitlines()) == 2
    trust = run["trust"]
    review = trust["requirementAcceptance"]
    assert review["required"] and review["status"] == "pass", review
    assert trust["completionStatus"] == "completed" and run["success"], trust
    assert not any(call["toolName"].startswith(("browser_", "desktop_")) for call in run["toolCalls"])
    request.node.live_passed = True
