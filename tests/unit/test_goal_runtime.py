"""Tests for rune.agent.goal_runtime - the runtime bridge.

Key regression: post_process runs exactly once per /goal loop (at the
terminal outcome), never per iteration, so a failing loop does not record
many negative episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.agent.goal_loop import GoalLoop, GoalLoopConfig, GoalSpec
from rune.agent.goal_runtime import GoalRuntime

# ---------------------------------------------------------------------------
# Fakes mirroring the real stack's shape
# ---------------------------------------------------------------------------


@dataclass
class FakeTrace:
    reason: str = "completed"
    final_step: int = 1
    total_tokens_used: int = 100
    evidence_score: float = 1.0


@dataclass
class FakeCtx:
    goal: str
    workspace_root: str = "/ws"


@dataclass
class FakeOpts:
    goal: str
    channel: str = "tui"
    conversation_id: str = ""


@dataclass
class FakePostInput:
    context: Any
    success: bool
    answer: str = ""


class FakeLoop:
    """Minimal EventEmitter + run. Emits scripted text_delta chunks so the
    runtime can capture an answer; records run kwargs for assertions."""

    def __init__(self, traces: list[FakeTrace], chunks: list[str] | None = None) -> None:
        self._traces = traces
        self._chunks = chunks or ["partial ", "answer"]
        self._handlers: dict[str, list[Any]] = {}
        self.run_kwargs: list[dict[str, Any]] = []
        self._n = 0

    def on(self, event: str, handler: Any) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Any) -> None:
        self._handlers.get(event, []).remove(handler)

    async def run(self, goal: str, **kwargs: Any) -> FakeTrace:
        self.run_kwargs.append({"goal": goal, **kwargs})
        for h in list(self._handlers.get("text_delta", [])):
            for c in self._chunks:
                h(c)
        tr = self._traces[min(self._n, len(self._traces) - 1)]
        self._n += 1
        return tr


class PostSpy:
    def __init__(self) -> None:
        self.calls: list[FakePostInput] = []

    async def __call__(self, inp: FakePostInput) -> None:
        self.calls.append(inp)


@dataclass
class PrepareSpy:
    seen: list[FakeOpts] = field(default_factory=list)

    async def __call__(self, opts: FakeOpts) -> FakeCtx:
        self.seen.append(opts)
        return FakeCtx(goal=opts.goal)


def _runtime(loop: FakeLoop, post: PostSpy, prep: PrepareSpy) -> GoalRuntime:
    return GoalRuntime(
        loop,
        prepare=prep,
        post_process=post,
        make_opts=FakeOpts,
        make_post_input=FakePostInput,
    )


def _goal_loop(rt: GoalRuntime, cfg: GoalLoopConfig, **kw: Any) -> GoalLoop:
    return GoalLoop(
        cfg,
        run_fn=rt.run_fn,
        persist_fn=rt.persist_fn,
        answer_of=rt.answer_of,
        **kw,
    )


def spec() -> GoalSpec:
    return GoalSpec(goal="do X", acceptance_criteria=["ac"])


# ---------------------------------------------------------------------------
# Poisoning-safe regression
# ---------------------------------------------------------------------------


async def test_post_process_once_on_success(tmp_path) -> None:
    loop = FakeLoop([FakeTrace(reason="", evidence_score=0.0)] * 2 + [FakeTrace()])
    post, prep = PostSpy(), PrepareSpy()
    gl = _goal_loop(
        _runtime(loop, post, prep),
        GoalLoopConfig(stagnation_window=0),
        workspace=tmp_path,
    )

    res = await gl.run(spec())

    assert res.success is True
    assert len(res.iterations) == 3
    assert len(post.calls) == 1  # terminal-only, not 3
    assert post.calls[0].success is True
    assert post.calls[0].answer == "partial answer"


async def test_post_process_once_on_failure(tmp_path) -> None:
    loop = FakeLoop([FakeTrace(reason="", evidence_score=0.0)])
    post, prep = PostSpy(), PrepareSpy()
    gl = _goal_loop(
        _runtime(loop, post, prep),
        GoalLoopConfig(max_iterations=5, stagnation_window=0),
        workspace=tmp_path,
    )

    res = await gl.run(spec())

    assert res.success is False
    assert len(res.iterations) == 5
    assert len(post.calls) == 1  # 5 failed attempts -> one neutral persistence
    assert post.calls[0].success is False


async def test_fresh_context_every_iteration(tmp_path) -> None:
    loop = FakeLoop([FakeTrace(reason="", evidence_score=0.0), FakeTrace()])
    post, prep = PostSpy(), PrepareSpy()
    gl = _goal_loop(
        _runtime(loop, post, prep),
        GoalLoopConfig(stagnation_window=0),
        workspace=tmp_path,
    )

    await gl.run(spec())

    # No chat history carried across attempts (anti Context-Rot), no resume.
    assert all(rk["message_history"] is None for rk in loop.run_kwargs)
    assert all("resume_session_id" not in rk for rk in loop.run_kwargs)
    assert len(prep.seen) == 2  # context re-prepared each attempt


async def test_text_delta_handler_not_leaked(tmp_path) -> None:
    loop = FakeLoop([FakeTrace(reason="", evidence_score=0.0), FakeTrace()])
    post, prep = PostSpy(), PrepareSpy()
    gl = _goal_loop(
        _runtime(loop, post, prep),
        GoalLoopConfig(stagnation_window=0),
        workspace=tmp_path,
    )

    await gl.run(spec())

    # Each run_fn registers then removes its listener (off in finally).
    assert loop._handlers.get("text_delta", []) == []


async def test_cancelled_before_start_skips_post_process(tmp_path) -> None:
    loop = FakeLoop([FakeTrace()])
    post, prep = PostSpy(), PrepareSpy()
    gl = _goal_loop(
        _runtime(loop, post, prep),
        GoalLoopConfig(),
        cancelled=lambda: True,
        workspace=tmp_path,
    )

    res = await gl.run(spec())

    assert res.stop_cause == "cancelled"
    assert post.calls == []  # nothing ran -> no episodic write at all
    assert loop.run_kwargs == []


# ---------------------------------------------------------------------------
# Phase 5.1/5.5: source manifest (always include project source; mark
# changed-this-iteration vs baseline; reviewer always sees the real code)
# ---------------------------------------------------------------------------

import os as _os  # noqa: E402

from rune.agent.goal_runtime import _collect_artifact  # noqa: E402


def _mk(p, text: str) -> None:
    p.write_text(text, encoding="utf-8")


async def test_manifest_lists_all_marks_shown_when_small(tmp_path) -> None:
    _mk(tmp_path / "a.go", "package x\nfunc A(){}\n")
    _mk(tmp_path / "b.go", "package x\nfunc B(){}\n")
    out = _collect_artifact(
        str(tmp_path), 0.0, set(), max_total=16384, per_file=4096, max_files=40
    )
    assert "SOURCE MANIFEST - 2 file(s), 2 with content shown" in out
    assert "- a.go (2 lines," in out and "[shown]" in out
    assert "=== a.go ===" in out and "=== b.go ===" in out


async def test_manifest_marks_omitted_when_over_file_cap(tmp_path) -> None:
    for i in range(4):
        _mk(tmp_path / f"f{i}.go", f"package x\n// file {i}\n")
    out = _collect_artifact(
        str(tmp_path), 0.0, set(), max_total=16384, per_file=4096, max_files=2
    )
    assert "2 with content shown" in out
    assert out.count("[shown]") == 2
    assert "[omitted: cap]" in out  # exists in manifest but body not included


async def test_excludes_lock_dot_dirs_and_state_files(tmp_path) -> None:
    _mk(tmp_path / "go.sum", "h1:abc\n")
    (tmp_path / ".rune").mkdir()
    _mk(tmp_path / ".rune" / "progress.md", "secret\n")
    _mk(tmp_path / "feedback.md", "worker-written validation log\n")  # root!
    _mk(tmp_path / "main.go", "package main\n")
    out = _collect_artifact(
        str(tmp_path),
        0.0,
        set(),
        max_total=16384,
        per_file=4096,
        max_files=40,
        exclude_names=frozenset({"feedback.md", "progress.md", "SPEC.md"}),
    )
    assert "go.sum" not in out  # suffix-excluded
    assert "feedback.md" not in out  # 5.5: root state file basename-excluded
    assert "progress.md" not in out  # .rune-dir + basename
    assert "main.go" in out


async def test_baseline_unchanged_source_still_included(tmp_path) -> None:
    # 5.5 core: a file NOT changed this iteration must still be shown
    # (tagged [baseline]) so the reviewer keeps seeing the real code; the
    # changed file is listed first.
    old = tmp_path / "lib.rs"
    _mk(old, "fn solution() {}\n")
    _os.utime(old, (1000, 1000))  # long before started_at -> baseline
    churn = tmp_path / "notes.txt"
    _mk(churn, "iteration churn\n")
    _os.utime(churn, (9_000_000_000, 9_000_000_000))  # after -> changed
    out = _collect_artifact(
        str(tmp_path), 5_000_000_000.0, set(),
        max_total=16384, per_file=4096, max_files=40,
    )
    assert "- lib.rs (1 lines," in out and "[baseline]" in out
    assert "- notes.txt" in out and "[changed]" in out
    assert "=== lib.rs ===" in out  # real code visible despite being unchanged
    assert out.index("notes.txt") < out.index("lib.rs")  # changed first


async def test_no_candidates_returns_empty(tmp_path) -> None:
    # empty workspace -> nothing to scan -> empty artifact
    out = _collect_artifact(
        str(tmp_path), 9e18, set(), max_total=16384, per_file=4096, max_files=40
    )
    assert out == ""
