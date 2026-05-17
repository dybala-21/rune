"""Outer loop for the ``/goal`` command.

The inner NativeAgentLoop already injects unmet completion-gate requirements
and continues until it reports verified or a budget/stall exit. This module
adds a separate fresh context for each attempt, carrying state only through
files (SPEC, fix_plan, progress) so a long-running goal does not accumulate
stale context.

See ``docs/design/goal-command.md``. The core has no UI dependency and takes
its collaborators by injection (run / validate / persist / cancel) so it can
be unit-tested with stub traces without importing the agent stack.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from rune.agent.obs_cap import head_tail
from rune.utils.logger import get_logger

log = get_logger(__name__)

Verdict = Literal["verified", "progress", "stagnant", "error", "cancelled"]
StopCause = Literal[
    "verified", "max_iterations", "budget", "stagnation", "cancelled", "error"
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GoalLoopConfig:
    max_iterations: int = 10
    max_total_tokens: int = 2_000_000  # cap on cumulative token spend
    stagnation_window: int = 3  # identical outcomes in a row -> stop
    evidence_threshold: float = 0.8  # inner-loop evidence_score floor
    adversarial_review: bool = False  # run the allow/block gate before accepting
    ssc_interval: int = 0  # self-critique every N iterations (0 = off)
    spec_file: str = "SPEC.md"
    plan_file: str = "fix_plan.md"
    progress_file: str = "progress.md"
    feedback_file: str = "feedback.md"  # latest failure only (overwritten)
    feedback_file_max: int = 4096  # cap when the loop writes feedback.md
    feedback_inject_chars: int = 1200  # cap for the prompt-injected excerpt


@dataclass(slots=True)
class GoalSpec:
    """Immutable acceptance spec the loop is held to for every iteration."""

    goal: str
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoalIteration:
    n: int
    reason: str
    evidence: float
    tokens: int
    verdict: Verdict
    validation_passed: bool | None = None
    review_passed: bool | None = None


@dataclass(slots=True)
class GoalLoopResult:
    success: bool
    stop_cause: StopCause
    iterations: list[GoalIteration] = field(default_factory=list)
    final_answer: str = ""


# ---------------------------------------------------------------------------
# Injected dependency contracts
# ---------------------------------------------------------------------------


class TraceLike(Protocol):
    """Subset of :class:`rune.types.CompletionTrace` the loop reads."""

    reason: str
    final_step: int
    total_tokens_used: int
    evidence_score: float


RunFn = Callable[[str, int], Awaitable[TraceLike]]
"""``(prompt, iteration) -> trace``. Wraps ``NativeAgentLoop.run``."""

ValidateFn = Callable[[list[str]], Awaitable[tuple[bool, str]]]
"""``(commands) -> (passed, detail)``. Runs SPEC validation deterministically."""

PersistFn = Callable[[bool, str], Awaitable[None]]
"""``(success, answer) -> None``. Persists episodic memory once, at the
terminal outcome only, so repeated failed attempts are not recorded."""


@dataclass(slots=True, frozen=True)
class ReviewContext:
    """Structured input to the review gate, like ``CompletionGateInput``."""

    spec: GoalSpec
    claim: str  # agent prose; not trusted, kept as context only
    validation_output: str  # deterministic command transcript
    artifact: str = ""  # bounded snapshot of changed source (may be "")


ReviewFn = Callable[["ReviewContext"], Awaitable[tuple[bool, str]]]
"""``(ReviewContext) -> (allow, reason)``. Reviews a verified candidate using
the deterministic validation output and the changed source, so a passing but
empty/no-assertion test is rejected. On error, returns block."""

CritiqueFn = Callable[["GoalSpec", str, int], Awaitable[str]]
"""``(spec, answer, iteration) -> critique``. Advisory, does not block."""


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


class GoalLoop:
    """Fresh-context outer loop. See module docstring / design doc."""

    def __init__(
        self,
        config: GoalLoopConfig | None = None,
        *,
        run_fn: RunFn,
        validate_fn: ValidateFn | None = None,
        persist_fn: PersistFn | None = None,
        cancelled: Callable[[], bool] | None = None,
        answer_of: Callable[[TraceLike], str] | None = None,
        on_iteration: Callable[[GoalIteration], None] | None = None,
        review_fn: ReviewFn | None = None,
        critique_fn: CritiqueFn | None = None,
        artifact_fn: Callable[[], Awaitable[str]] | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self._cfg = config or GoalLoopConfig()
        self._run = run_fn
        self._validate = validate_fn
        self._persist = persist_fn
        self._cancelled = cancelled or (lambda: False)
        self._answer_of = answer_of or (lambda t: getattr(t, "answer", "") or "")
        self._on_iteration = on_iteration
        self._review = review_fn
        self._critique = critique_fn
        self._artifact_fn = artifact_fn
        self._ws = Path(workspace) if workspace else None
        self._last_feedback = ""  # bounded failure excerpt for the next prompt

    def _record(self, hist: list[GoalIteration], it: GoalIteration) -> None:
        """Append an iteration and fire the (UI-only) progress callback."""
        hist.append(it)
        if self._on_iteration is not None:
            try:
                self._on_iteration(it)
            except Exception as exc:  # a render error should not stop the loop
                log.debug("goal_loop_on_iteration_failed", error=str(exc)[:200])

    # -- verdict -----------------------------------------------------------

    def _verdict(self, trace: TraceLike) -> Verdict:
        # A "completed" inner run is only a candidate. The deterministic
        # validation commands and the review gate decide acceptance. The inner
        # loop often reports evidence_score=0.0 even on success, so evidence is
        # not used as a pre-gate here.
        reason = trace.reason or ""
        if reason == "completed":
            return "verified"
        if reason == "cancelled":
            return "cancelled"
        if reason.startswith("error:"):
            return "error"
        return "progress"

    def _is_stagnant(self, hist: list[GoalIteration]) -> bool:
        """True when the last ``window`` iterations produced an identical
        outcome signature (reason, evidence, tokens)."""
        w = self._cfg.stagnation_window
        if w <= 0 or len(hist) < w:
            return False
        recent = hist[-w:]

        def _sig(it: GoalIteration) -> tuple[str, float, int]:
            return (it.reason, round(it.evidence, 2), it.tokens)

        first = _sig(recent[0])
        return all(_sig(it) == first for it in recent)

    # -- workspace state files (best-effort, does not crash the loop) ------

    def _seed_files(self, spec: GoalSpec) -> None:
        if self._ws is None:
            return
        try:
            self._ws.mkdir(parents=True, exist_ok=True)
            spec_path = self._ws / self._cfg.spec_file
            if not spec_path.exists():  # immutable anchor: write once only
                ac = "\n".join(f"- {c}" for c in spec.acceptance_criteria)
                con = "\n".join(f"- {c}" for c in spec.constraints)
                cmds = "\n".join(f"- `{c}`" for c in spec.validation_commands)
                spec_path.write_text(
                    f"# SPEC (immutable)\n\n## Goal\n{spec.goal}\n\n"
                    f"## Acceptance Criteria\n{ac}\n\n## Constraints\n{con}\n\n"
                    f"## Validation Commands\n{cmds}\n",
                    encoding="utf-8",
                )
            plan_path = self._ws / self._cfg.plan_file
            if not plan_path.exists():
                items = "\n".join(f"- [ ] {c}" for c in spec.acceptance_criteria) or "- [ ] (derive)"
                plan_path.write_text(f"# fix_plan - {spec.goal}\n\n{items}\n", encoding="utf-8")
            prog = self._ws / self._cfg.progress_file
            if not prog.exists():
                prog.write_text(f"# progress - {spec.goal}\n\n## Avoid\n\n## Log\n", encoding="utf-8")
        except OSError as exc:
            log.debug("goal_loop_seed_files_failed", error=str(exc)[:200])

    def _append_progress(self, line: str) -> None:
        if self._ws is None:
            return
        try:
            with (self._ws / self._cfg.progress_file).open("a", encoding="utf-8") as fh:
                fh.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
        except OSError as exc:
            log.debug("goal_loop_progress_append_failed", error=str(exc)[:200])

    # -- failure feedback reflux ------------------------------------------

    def _feedback_path(self) -> Path | None:
        return None if self._ws is None else self._ws / self._cfg.feedback_file

    def _feedback_stat(self) -> tuple[float, int] | None:
        p = self._feedback_path()
        try:
            if p is not None and p.is_file():
                st = p.stat()
                return (st.st_mtime, st.st_size)
        except OSError:
            pass
        return None

    def _write_feedback(self, kind: str, text: str) -> None:
        """Overwrite feedback.md with the latest failure (single-latest, not
        appended) and cache a bounded excerpt for the next prompt."""
        body = head_tail(text or "", self._cfg.feedback_file_max)
        self._last_feedback = head_tail(
            f"[{kind}]\n{text or ''}", self._cfg.feedback_inject_chars
        )
        p = self._feedback_path()
        if p is None:
            return
        try:
            p.write_text(
                f"# feedback (latest failure only) - {kind}\n\n{body}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            log.debug("goal_loop_feedback_write_failed", error=str(exc)[:200])

    def _adopt_worker_feedback(self) -> bool:
        """If the worker overwrote feedback.md itself this iteration, use it
        as the next prompt's excerpt. Returns True when adopted."""
        p = self._feedback_path()
        try:
            if p is not None and p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    self._last_feedback = head_tail(
                        content, self._cfg.feedback_inject_chars
                    )
                    return True
        except OSError as exc:
            log.debug("goal_loop_feedback_read_failed", error=str(exc)[:200])
        return False

    def _feedback_rel(self) -> str:
        """Path the worker should overwrite with its validation output. Points
        at the loop's own feedback file (inside the state dir, which the
        reviewer artifact excludes) so the worker does not drop a markdown
        file into the project root and pose it as source."""
        if self._ws is None:
            return self._cfg.feedback_file
        try:
            return os.path.relpath(
                self._ws / self._cfg.feedback_file, os.getcwd()
            )
        except (OSError, ValueError):
            return self._cfg.feedback_file

    def _build_prompt(self, spec: GoalSpec, n: int) -> str:
        prefix = ""
        if n > 1 and self._last_feedback:
            prefix = (
                "[PREVIOUS ATTEMPT FAILED - fix exactly this before anything "
                f"else]\n{self._last_feedback}\n\n"
            )
        return (
            f"{prefix}{spec.goal}\n\n"
            f"Work ONE unfinished item from @{self._cfg.plan_file} this iteration "
            f"(iteration {n}). Honour the immutable @{self._cfg.spec_file}. When the "
            f"item is done, check it off in @{self._cfg.plan_file} and append a "
            f"one-line learning to @{self._cfg.progress_file}. Before finishing, "
            f"run the project's validation commands and overwrite "
            f"{self._feedback_rel()} with their full output. "
            f"Do not start unrelated work."
        )

    # -- main --------------------------------------------------------------

    async def run(self, spec: GoalSpec) -> GoalLoopResult:
        if (
            spec.acceptance_criteria
            and not spec.validation_commands
            and not (self._cfg.adversarial_review and self._review is not None)
        ):
            log.warning(
                "goal_loop_no_objective_check",
                detail="no validation_commands and no reviewer; accept will "
                "fall back to the (often 0.0) inner evidence score",
            )
        self._seed_files(spec)
        hist: list[GoalIteration] = []
        total_tokens = 0
        last_answer = ""
        stop_cause: StopCause = "max_iterations"

        for n in range(1, self._cfg.max_iterations + 1):
            if self._cancelled():
                stop_cause = "cancelled"
                break

            fb_pre = self._feedback_stat()
            wrote_fb = False
            try:
                trace = await self._run(self._build_prompt(spec, n), n)
            except Exception as exc:  # on a run error, end the loop
                log.warning("goal_loop_run_error", iteration=n, error=str(exc)[:200])
                stop_cause = "error"
                self._record(hist, GoalIteration(n, f"error: {exc}"[:120], 0.0, 0, "error"))
                break

            total_tokens += getattr(trace, "total_tokens_used", 0) or 0
            last_answer = self._answer_of(trace) or last_answer
            verdict = self._verdict(trace)
            it = GoalIteration(
                n=n,
                reason=trace.reason or "",
                evidence=float(getattr(trace, "evidence_score", 0.0) or 0.0),
                tokens=int(getattr(trace, "total_tokens_used", 0) or 0),
                verdict=verdict,
            )

            if verdict == "verified":
                # With no validation commands and no reviewer there is no
                # objective check, so fall back to the inner evidence score
                # rather than accept a bare "completed".
                no_objective = not spec.validation_commands and not (
                    self._cfg.adversarial_review and self._review is not None
                )
                if no_objective and it.evidence < self._cfg.evidence_threshold:
                    it.verdict = "progress"
                    self._append_progress(
                        f"Avoid: 'completed' unsubstantiated (evidence "
                        f"{it.evidence:.2f} < {self._cfg.evidence_threshold})"
                    )
                else:
                    passed, detail = await self._run_validation(spec)
                    it.validation_passed = passed
                    if passed:
                        # Review the deterministic validation output, not the
                        # agent's prose claim (last_answer).
                        allow, why = await self._adversarial_review(
                            spec, last_answer, detail
                        )
                        it.review_passed = allow
                        if allow:
                            self._record(hist, it)
                            stop_cause = "verified"
                            break
                        # Review blocked a test-passing candidate.
                        it.verdict = "progress"
                        self._append_progress(
                            f"Avoid: adversarial review blocked: {why[:200]}"
                        )
                        self._write_feedback("adversarial-review-block", why)
                        wrote_fb = True
                    else:
                        # Inner loop said done but validation fails.
                        it.verdict = "progress"
                        self._append_progress(
                            f"Avoid: validation failed: {detail[:200]}"
                        )
                        self._write_feedback("validation-failed", detail)
                        wrote_fb = True
            elif verdict in ("cancelled", "error"):
                self._record(hist, it)
                stop_cause = verdict
                break

            # Feedback reflux: an attempt that did not reach the loop's own
            # validation (e.g. token_budget_exhausted). Adopt the worker's
            # self-captured feedback.md if it overwrote it this iteration;
            # otherwise leave a directive so the next attempt is steered.
            if not wrote_fb and it.verdict == "progress":
                changed = self._feedback_stat() != fb_pre
                if not (changed and self._adopt_worker_feedback()):
                    self._write_feedback(
                        "no-validation",
                        f"previous attempt did not reach validation "
                        f"(reason={trace.reason or 'unknown'}); reach the SPEC "
                        f"validation commands faster, avoid over-exploration",
                    )

            self._record(hist, it)
            await self._ssc_critique(spec, last_answer, n)

            if total_tokens > self._cfg.max_total_tokens:
                stop_cause = "budget"
                break
            if self._is_stagnant(hist):
                stop_cause = "stagnation"
                break
            # otherwise continue with a new context

        success = stop_cause == "verified"
        # Persist episodic memory once, at the terminal outcome only.
        if self._persist is not None:
            try:
                await self._persist(success, last_answer)
            except Exception as exc:
                log.debug("goal_loop_persist_failed", error=str(exc)[:200])
        self._append_progress(
            f"DONE stop_cause={stop_cause} success={success} iterations={len(hist)}"
        )
        return GoalLoopResult(
            success=success,
            stop_cause=stop_cause,
            iterations=hist,
            final_answer=last_answer,
        )

    async def _run_validation(self, spec: GoalSpec) -> tuple[bool, str]:
        """The loop runs the SPEC validation commands itself. With no commands
        or no validator, the inner gate verdict is trusted."""
        if not spec.validation_commands or self._validate is None:
            return True, "no validation commands"
        try:
            return await self._validate(spec.validation_commands)
        except Exception as exc:
            log.debug("goal_loop_validate_error", error=str(exc)[:200])
            return False, f"validator error: {exc}"[:200]

    async def _adversarial_review(
        self, spec: GoalSpec, answer: str, evidence: str
    ) -> tuple[bool, str]:
        """Allow/block a validation-passing candidate. Returns allow unless
        the config opts in and a reviewer is injected. Builds a ReviewContext
        (validation output plus changed source) so the reviewer can reject a
        passing but empty/no-assertion test. Returns block on reviewer error."""
        if not self._cfg.adversarial_review or self._review is None:
            return True, "review disabled"
        artifact = ""
        if self._artifact_fn is not None:
            try:  # context only; a gather failure should not block on its own
                artifact = await self._artifact_fn() or ""
            except Exception as exc:
                log.debug("goal_loop_artifact_failed", error=str(exc)[:200])
        ctx = ReviewContext(
            spec=spec,
            claim=answer,
            validation_output=evidence,
            artifact=artifact,
        )
        try:
            allow, reason = await self._review(ctx)
            return bool(allow), str(reason or "")
        except Exception as exc:  # do not accept when the reviewer fails
            log.debug("goal_loop_review_error", error=str(exc)[:200])
            return False, f"reviewer error: {exc}"[:200]

    async def _ssc_critique(self, spec: GoalSpec, answer: str, n: int) -> None:
        """Advisory self-critique every ``ssc_interval`` iterations. Records a
        note to progress.md; does not block the loop."""
        iv = self._cfg.ssc_interval
        if iv <= 0 or self._critique is None or n % iv != 0:
            return
        try:
            note = await self._critique(spec, answer, n)
        except Exception as exc:
            log.debug("goal_loop_ssc_error", error=str(exc)[:200])
            return
        if note and note.strip():
            self._append_progress(f"SSC (iter {n}): {note.strip()[:300]}")
