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


# F1/F2/F3: build-output dirs must not starve real source out of the
# reviewer artifact (a target/ flood once buried Cargo.toml).


async def test_build_output_dir_excluded_and_manifest_forced(tmp_path) -> None:
    _mk(tmp_path / "Cargo.toml", '[package]\nname = "x"\nedition = "2021"\n')
    (tmp_path / "src").mkdir()
    _mk(tmp_path / "src" / "lib.rs", "fn main() {}\n")
    tgt = tmp_path / "target" / "debug" / ".fingerprint"
    tgt.mkdir(parents=True)
    (tmp_path / "target" / "CACHEDIR.TAG").write_text("Signature\n")
    for i in range(200):  # the flood
        _mk(tgt / f"junk{i}.json", '{"rustc":' + str(i) + "}\n")
    _os.utime(tmp_path / "Cargo.toml", (1000, 1000))  # unchanged -> baseline
    out = _collect_artifact(
        str(tmp_path), 5_000_000_000.0, set(),
        max_total=16384, per_file=4096, max_files=40,
    )
    assert "target/" not in out and "junk" not in out  # flood excluded
    assert "=== Cargo.toml ===" in out  # forced past the cap despite baseline
    assert "name = \"x\"" in out
    assert "=== src/lib.rs ===" in out


async def test_cachedir_tag_prunes_unknown_named_cache_dir(tmp_path) -> None:
    # a cache dir whose name is NOT in _EXCLUDE_DIRS is still pruned by the
    # CACHEDIR.TAG content marker (no per-tool name maintenance).
    _mk(tmp_path / "main.go", "package main\n")
    weird = tmp_path / "weirdcache"
    weird.mkdir()
    (weird / "CACHEDIR.TAG").write_text("Signature: ...\n")
    _mk(weird / "huge.go", "package x\n" + "// junk\n" * 500)
    out = _collect_artifact(
        str(tmp_path), 0.0, set(), max_total=16384, per_file=4096, max_files=40
    )
    assert "main.go" in out
    assert "weirdcache" not in out and "huge.go" not in out


async def test_manifest_file_forced_past_file_cap(tmp_path) -> None:
    # non-forced files exhaust the file cap; the project-definition file is
    # still shown (its criteria must stay verifiable) and listed first.
    _mk(tmp_path / "Cargo.toml", '[package]\nname = "rust_webrtc"\n')
    for i in range(6):
        _mk(tmp_path / f"s{i}.rs", f"// source {i}\n")
    out = _collect_artifact(
        str(tmp_path), 0.0, set(), max_total=16384, per_file=4096, max_files=2
    )
    assert "=== Cargo.toml ===" in out and 'name = "rust_webrtc"' in out
    assert "[omitted: cap]" in out  # some non-forced gated out by max_files
    assert out.index("Cargo.toml") < out.index("s0.rs")  # forced listed first


async def test_tracked_outranks_newer_mtime_file(tmp_path) -> None:
    # the agent-written file (tracked) must rank above a non-tracked file
    # with a newer mtime, so regenerated artifacts never preempt real edits.
    edited = tmp_path / "edited.go"
    _mk(edited, "package x // agent wrote this\n")
    _os.utime(edited, (1000, 1000))  # OLD mtime
    regen = tmp_path / "regen.go"
    _mk(regen, "package y // build regenerated\n")
    _os.utime(regen, (9_000_000_000, 9_000_000_000))  # NEW mtime
    out = _collect_artifact(
        str(tmp_path), 0.0, {str(edited)},
        max_total=16384, per_file=4096, max_files=40,
    )
    assert out.index("edited.go") < out.index("regen.go")


# F5: a small project's changed/definition files must be shown whole;
# a too-small per-file cap once mid-elided the only source file.

_F5 = dict(max_total=65536, per_file=24576, per_file_baseline=6144)  # shipped


async def test_changed_source_shown_whole_no_mid_elision(tmp_path) -> None:
    body = "fn a() {}\n" + "// signaling wiring\n" * 500 + "fn END_MARK() {}\n"
    assert 5000 < len(body) < 24576  # realistic single source file
    _mk(tmp_path / "lib.rs", body)
    out = _collect_artifact(
        str(tmp_path), 0.0, {str(tmp_path / "lib.rs")}, max_files=40, **_F5
    )
    assert "fn END_MARK() {}" in out  # tail present -> not middle-elided
    assert "elided" not in out  # head_tail untouched (limit >= file size)


async def test_baseline_capped_tighter_than_changed(tmp_path) -> None:
    # marker sits at the CENTRE so the head/tail-keeping elision drops it
    # only when the (smaller) baseline limit applies.
    base = tmp_path / "old.rs"  # ~9009 chars, "BASE_MID" at the centre
    _mk(base, "H" * 4500 + "BASE_MID" + "T" * 4500)
    _os.utime(base, (1000, 1000))  # baseline
    chg = tmp_path / "new.rs"
    _mk(chg, "H" * 4500 + "CHG_MID" + "T" * 4500)
    _os.utime(chg, (9_000_000_000, 9_000_000_000))  # changed
    out = _collect_artifact(
        str(tmp_path), 5_000_000_000.0, set(), max_files=40, **_F5
    )
    assert "CHG_MID" in out  # changed: whole at per_file=24576
    assert "BASE_MID" not in out and "elided" in out  # baseline elided at 6144


async def test_max_total_ceiling_still_bounds(tmp_path) -> None:
    for i in range(8):  # 8 x ~20KB changed files, all whole would be ~160KB
        _mk(tmp_path / f"big{i}.rs", f"// {i}\n" + "z" * 20000)
    out = _collect_artifact(
        str(tmp_path), 0.0, set(), max_files=40, **_F5
    )
    body = out.split("\n\n", 1)[1]
    assert len(body) <= _F5["max_total"]  # hard ceiling holds
    assert "[omitted: cap]" in out  # excess files bounded out
