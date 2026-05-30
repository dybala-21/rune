from __future__ import annotations

from pytest import MonkeyPatch

from rune.agent.loop import NativeAgentLoop, _failed_tool_nudge, _tool_result_event_payload
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


def test_benchmark_completion_blocker_requires_recent_failure_and_env(
    monkeypatch: MonkeyPatch,
) -> None:
    loop = NativeAgentLoop()
    loop._recent_failed_tool_nudge = "failed command output"

    monkeypatch.delenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", raising=False)
    assert loop._benchmark_completion_blocker() is None

    monkeypatch.setenv("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT", "1")
    blocker = loop._benchmark_completion_blocker()

    assert blocker is not None
    assert "recent tool failed" in blocker
    assert "failed command output" in blocker
