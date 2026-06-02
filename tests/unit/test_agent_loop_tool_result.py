from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from rune.agent.loop import (
    NativeAgentLoop,
    _failed_tool_nudge,
    _should_clear_failed_tool_nudge,
    _tool_result_event_payload,
)
from rune.types import CapabilityResult


def test_failed_tool_result_tail_capture_is_env_gated(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", raising=False)

    payload = _tool_result_event_payload(
        "bash_execute",
        CapabilityResult(success=False, output="build failed", error="exit 1"),
    )

    assert payload["output_length"] == len("build failed")
    assert "output_tail" not in payload
    assert "error_tail" not in payload


def test_failed_tool_result_tail_capture_is_bounded(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", "1")
    monkeypatch.setenv("RUNE_BENCH_FAILED_TOOL_OUTPUT_TAIL_BYTES", "8")

    payload = _tool_result_event_payload(
        "bash_execute",
        CapabilityResult(success=False, output="0123456789abcdef", error="permission denied"),
    )

    assert payload["output_tail"] == "89abcdef"
    assert payload["output_tail_truncated"] is True
    assert payload["error_tail"] == "n denied"


def test_successful_tool_result_does_not_capture_output_tail(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", "1")

    payload = _tool_result_event_payload(
        "bash_execute",
        CapabilityResult(success=True, output="ok"),
    )

    assert payload["output_length"] == 2
    assert "output_tail" not in payload


def test_failed_tool_nudge_uses_concrete_error_and_output_tail() -> None:
    nudge = _failed_tool_nudge(
        "bash_execute",
        CapabilityResult(success=False, output="first line\nlast line", error="exit 2"),
    )

    assert "Most recent failed tool result" in nudge
    assert "bash_execute" in nudge
    assert "exit 2" in nudge
    assert "last line" in nudge


@pytest.mark.asyncio
async def test_benchmark_completion_blocker_requires_recent_failure_and_env(
    monkeypatch: MonkeyPatch,
) -> None:
    loop = NativeAgentLoop()
    loop._recent_failed_tool_nudge = "failed command output"

    monkeypatch.delenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", raising=False)
    assert await loop._benchmark_completion_blocker() is None

    monkeypatch.setenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", "1")
    blocker = await loop._benchmark_completion_blocker()

    assert blocker is not None
    assert "recent tool failed" in blocker
    assert "failed command output" in blocker


class _FakeGate:
    def __init__(self, state: str, message: str | None = None):
        self._state = state
        self._message = message

    async def verdict(self):
        return self._state, self._message

    def summary(self):
        return {"last_verdict": self._state}


@pytest.mark.asyncio
async def test_evidence_verdict_skip_when_no_gate(monkeypatch: MonkeyPatch) -> None:
    loop = NativeAgentLoop()
    loop._evidence_gate = None
    assert await loop._evidence_verdict() == ("skip", None)


@pytest.mark.asyncio
async def test_evidence_verdict_passes_through_states() -> None:
    loop = NativeAgentLoop()
    loop._evidence_gate = _FakeGate("pass")
    assert await loop._evidence_verdict() == ("pass", None)
    loop._evidence_gate = _FakeGate("fail", "first mismatch line 3")
    state, msg = await loop._evidence_verdict()
    assert state == "fail" and msg == "first mismatch line 3"


@pytest.mark.asyncio
async def test_evidence_verdict_swallows_errors() -> None:
    class _Boom:
        async def verdict(self):
            raise RuntimeError("boom")

    loop = NativeAgentLoop()
    loop._evidence_gate = _Boom()
    # An exception during verification must degrade to neutral "skip",
    # never propagate and never block a real success.
    assert await loop._evidence_verdict() == ("skip", None)


@pytest.mark.asyncio
async def test_blocker_returns_fail_message_from_gate(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", raising=False)
    loop = NativeAgentLoop()
    loop._recent_failed_tool_nudge = ""
    loop._evidence_gate = _FakeGate("fail", "[Evidence Gate] got X exp Y")
    blocker = await loop._benchmark_completion_blocker()
    assert blocker == "[Evidence Gate] got X exp Y"

    # A passing gate must NOT produce a blocker (lets finalize proceed).
    loop._evidence_gate = _FakeGate("pass")
    assert await loop._benchmark_completion_blocker() is None


def test_failed_tool_nudge_clears_after_successful_bash() -> None:
    assert _should_clear_failed_tool_nudge(
        "bash_execute",
        CapabilityResult(success=True, output="ok"),
    )
    assert not _should_clear_failed_tool_nudge(
        "bash_execute",
        CapabilityResult(success=False, error="still failed"),
    )
    assert not _should_clear_failed_tool_nudge(
        "file_write",
        CapabilityResult(success=True, output="wrote file"),
    )
