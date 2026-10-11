"""Calibrate the claim reviewer with synthetic code and real test output."""

import subprocess
import sys
import time

import pytest

from rune.agent.classification_response import decode_object
from rune.agent.test_claims import TestClaimGate as ClaimGate
from rune.agent.timing import capture_timing, timing_snapshot
from rune.agent.verification_state import VerificationState
from rune.types import CapabilityResult

pytestmark = pytest.mark.asyncio(loop_scope="module")
CODE = "def mean(values):\n    return sum(values) / (len(values) + 1)\n"
FIXED = "def mean(values):\n    if not values:\n        raise ValueError('empty input')\n    return sum(values) / len(values)\n"
TESTS = """import unittest
from mean import mean

class MeanTests(unittest.TestCase):
    def test_nonzero(self): self.assertEqual(mean([2, 4]), 3)
    def test_zero(self): self.assertEqual(mean([-2, 2]), 0)
    def test_empty(self):
        with self.assertRaises(ValueError): mean([])
"""


@pytest.fixture
def review_case(tmp_path):
    source = tmp_path / "mean.py"
    source.write_text(CODE)
    (tmp_path / "test_mean.py").write_text(TESTS)
    state, gate = VerificationState(), ClaimGate()
    for path, text in (("mean.py", CODE), ("test_mean.py", TESTS)):
        gate.observe("file_read", {"path": path}, CapabilityResult(success=True, output=text))
    for index in range(2):
        if index:
            source.write_text(FIXED)
            state.changed()
            gate.observe("file_edit", {"path": "mean.py", "old": CODE, "new": FIXED},
                         CapabilityResult(success=True, output="Saved mean.py"))
        command = "python -B -m unittest -v test_mean"
        run = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", "test_mean"],
                             cwd=tmp_path, capture_output=True, text=True, timeout=10)
        output = (run.stdout + run.stderr).replace(str(tmp_path), "/fixture")
        success = run.returncode == 0
        assert success == bool(index)
        state.observe_command(command, success, output, "/fixture")
        gate.observe("bash_execute", {"command": command, "cwd": "/fixture"},
                     CapabilityResult(success=success, output=output))
    return state, gate


@pytest.mark.parametrize("answer,needs_correction", [
    ("Before the fix, every input returned an incorrect mean.", True),
    ("분모가 하나 더 커서 모든 평균이 실제보다 작았습니다.", True),
    ("The old function divided by len(values) + 1. Empty input returned 0.0 instead of raising ValueError.", False),
    ("수정 전에는 1개 통과, 2개 실패했습니다. 수정 후에는 3개 모두 통과했습니다.", False),
], ids=["universal", "direction", "supported_behavior", "supported_counts"])
async def test_claim_review_scope(review_case, live_model, live_report, monkeypatch, request, answer, needs_correction):
    import rune.llm.client as clients

    client = clients.get_llm_client()
    replies = []

    class RecordingClient:
        async def completion(self, **kwargs):
            response = await client.completion(**kwargs)
            usage = response.get("usage")
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            replies.append({"result": decode_object(response), "usage": usage,
                            "finish_reason": response["choices"][0].get("finish_reason"),
                            "prompt_chars": sum(len(m["content"]) for m in kwargs["messages"])})
            return response

    monkeypatch.setattr(clients, "get_llm_client", lambda: RecordingClient())
    state, gate = review_case
    started = time.monotonic()
    with capture_timing() as timing:
        note = await gate.review(state, answer)
    seconds = time.monotonic() - started
    passed = (bool(note) == needs_correction and len(replies) == 1
              and (note is None or "contradicts or exceeds" in note))
    from scripts.e2e_provenance import background_run

    live_report({"outcome": "passed" if passed else "failed", "answer": answer,
                 "expected_correction": needs_correction, "scope": "Claim reviewer; not a full workflow",
                 "note": note, "replies": replies,
                 "runs": [background_run({"duration_ms": seconds * 1000, "timings": timing_snapshot(timing)})]})
    assert passed, note or replies
