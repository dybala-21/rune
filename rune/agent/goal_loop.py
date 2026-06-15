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

import asyncio
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

# Stuck outcomes: the model tried and could not pass validation. These warrant a
# stronger-model escalation; "cancelled" (user) and "error" (crash) do not.
_STUCK_STOP_CAUSES = frozenset({"stagnation", "max_iterations", "budget"})

# Inner-loop reasons that mean "ran out of room without self-reporting
# completed" - a weak model often burns its whole iteration budget here, so
# the next prompt is steered to do the minimum and validate.
_BUDGET_REASONS = frozenset(
    {"token_budget_exhausted", "max_gate_blocked", "stalled", "no_progress"}
)


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
    iteration_timeout_s: int = 1200  # in-loop watchdog (0 = off); best-effort
    max_transient_retries: int = 3  # recoverable inner errors before fatal
    max_extra_iterations: int = 10  # extra iters allowed while still advancing
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
    recoverable: bool = False  # a transient-error attempt (excluded from progress)
    fb_hash: int = 0  # hash of the feedback at record time (plateau detection)
    fb_kind: str = ""  # feedback category at record time (plateau detection)


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
        escalate_fn: RunFn | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self._cfg = config or GoalLoopConfig()
        self._run = run_fn
        # One final attempt on a stronger model when the local loop is stuck.
        # Injected (not None) only when the caller has opted in AND configured an
        # escalation path, so its mere presence is the consent to escalate.
        self._escalate = escalate_fn
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
        self._last_fb_kind = ""  # feedback category for plateau detection
        self._last_inner_reason = ""  # prior iter's inner reason (C1 steer)

    def _record(self, hist: list[GoalIteration], it: GoalIteration) -> None:
        """Append an iteration and fire the (UI-only) progress callback."""
        it.fb_hash = hash(self._last_feedback or "")
        it.fb_kind = self._last_fb_kind
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

    @staticmethod
    def _progress_rank(it: GoalIteration) -> int:
        """Ordinal distance toward 'done' (higher = closer). Deterministic,
        no LLM. Pure feedback churn does not raise this."""
        if it.review_passed:
            return 4
        if it.validation_passed is True:
            return 3
        if it.validation_passed is False:
            return 2
        return 1  # validation not even reached this iteration

    def _window(self, hist: list[GoalIteration]) -> list[GoalIteration]:
        """Last ``stagnation_window`` non-recoverable iterations (transient
        errors are infra noise: neither progress nor plateau)."""
        w = self._cfg.stagnation_window
        if w <= 0:
            return []
        real = [it for it in hist if not it.recoverable]
        return real[-w:] if len(real) >= w else []

    def _advancing(self, hist: list[GoalIteration]) -> bool:
        """Net ordinal improvement across the window (slow but converging)."""
        win = self._window(hist)
        return bool(win) and self._progress_rank(win[-1]) > self._progress_rank(
            win[0]
        )

    def _plateaued(self, hist: list[GoalIteration]) -> bool:
        """Truly stuck: window full, no ordinal advance, and the *kind* of
        failure is unchanged across the window. A repeated
        adversarial-review block is reworded run to run, so once
        validation is green yet review keeps rejecting on the same
        category that is a genuine plateau (the loop cannot satisfy the
        reviewer and should stop, not burn the whole budget). For other
        failures the excerpt must be identical too, so a model fixing a
        *different* error each iteration keeps its full budget."""
        win = self._window(hist)
        if not win or self._advancing(hist):
            return False
        if any(it.fb_kind != win[0].fb_kind for it in win):
            return False
        if win[0].fb_kind == "adversarial-review-block":
            return all(
                it.validation_passed is True and not it.review_passed
                for it in win
            )
        return all(it.fb_hash == win[0].fb_hash for it in win)

    @staticmethod
    def _err_sig(err: str) -> str:
        return (err or "").strip()[:120]

    @staticmethod
    def _classify_error(err: str) -> str:
        """transient (recoverable: infra/provider/protocol) vs fatal (our-code
        bug). Structured token/type match, no NL parsing. Unknown -> transient
        (the harness should absorb infra), but a consecutive identical error
        is treated as fatal by the caller."""
        e = (err or "").lower()
        fatal = (
            "attributeerror", "keyerror", "typeerror", "nameerror",
            "indexerror", "importerror", "modulenotfounderror",
            "notimplementederror", "assertionerror", "zerodivisionerror",
            "unboundlocalerror",
        )
        if any(t in e for t in fatal):
            return "fatal"
        return "transient"

    def _handle_inner_error(
        self,
        err: str,
        n: int,
        hist: list[GoalIteration],
        transient_retries: int,
        last_err_sig: str | None,
    ) -> tuple[str, int, str]:
        """An inner-run error: 'retry' (record as recoverable, capture feedback,
        next iteration runs fresh) or 'fatal' (stop). Fatal when the error is a
        our-code bug class, repeats identically back-to-back, or the transient
        budget is spent."""
        sig = self._err_sig(err)
        is_fatal = (
            self._classify_error(err) == "fatal"
            or sig == last_err_sig
            or transient_retries >= self._cfg.max_transient_retries
        )
        if is_fatal:
            self._record(
                hist, GoalIteration(n, f"error: {err}"[:120], 0.0, 0, "error")
            )
            return ("fatal", transient_retries, sig)
        self._record(
            hist,
            GoalIteration(
                n, f"error: {err}"[:120], 0.0, 0, "error", recoverable=True
            ),
        )
        self._write_feedback("transient-error", err)
        log.warning(
            "goal_loop_transient_recover",
            iteration=n,
            retries=transient_retries + 1,
            error=err[:160],
        )
        return ("retry", transient_retries + 1, sig)

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
        self._last_fb_kind = kind
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
        if n > 1 and self._last_inner_reason in _BUDGET_REASONS:
            cmds = ", ".join(spec.validation_commands) or "the validation commands"
            prefix += (
                "[BUDGET DISCIPLINE] The previous attempt spent its entire "
                "budget WITHOUT reaching validation. Do the MINIMUM needed to "
                f"make these pass, then STOP - no exploration, no refactor: "
                f"{cmds}\n\n"
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

        n = 0
        transient_retries = 0
        last_err_sig: str | None = None
        effective_max = self._cfg.max_iterations
        while n < effective_max:
            n += 1
            if self._cancelled():
                stop_cause = "cancelled"
                break

            fb_pre = self._feedback_stat()
            wrote_fb = False
            try:
                _coro = self._run(self._build_prompt(spec, n), n)
                if self._cfg.iteration_timeout_s > 0:
                    trace = await asyncio.wait_for(
                        _coro, timeout=self._cfg.iteration_timeout_s
                    )
                else:
                    trace = await _coro
            except Exception as exc:  # transient -> fresh retry; fatal -> stop
                action, transient_retries, last_err_sig = self._handle_inner_error(
                    f"{type(exc).__name__}: {exc}", n, hist,
                    transient_retries, last_err_sig,
                )
                if action == "fatal":
                    log.warning(
                        "goal_loop_run_error", iteration=n, error=str(exc)[:200]
                    )
                    stop_cause = "error"
                    break
                continue  # recovered: next iteration runs with a fresh context

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
            self._last_inner_reason = trace.reason or ""  # for the C1 steer

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
                    accepted, fb_written = await self._try_accept(
                        spec, it, last_answer, hist
                    )
                    if accepted:
                        self._record(hist, it)
                        stop_cause = "verified"
                        break
                    wrote_fb = wrote_fb or fb_written
            elif verdict == "cancelled":
                self._record(hist, it)
                stop_cause = "cancelled"
                break
            elif verdict == "error":
                action, transient_retries, last_err_sig = self._handle_inner_error(
                    it.reason, n, hist, transient_retries, last_err_sig
                )
                if action == "fatal":
                    stop_cause = "error"
                    break
                continue  # recovered: next iteration runs with a fresh context

            # C2: full decoupling. The loop's own deterministic validation -
            # not the inner self-report - is the arbiter. When the inner loop
            # did NOT say "completed" (token_budget_exhausted /
            # max_gate_blocked / stalled) but produced work, still run
            # validation: the on-disk result may already pass, so an
            # inner-gated iteration is not silently discarded. ``wrote_fb``
            # guards against re-validating the verified branch's own result.
            if (
                it.verdict == "progress"
                and not wrote_fb
                and not it.recoverable
                and (
                    bool(spec.validation_commands)
                    or (self._cfg.adversarial_review and self._review is not None)
                )
            ):
                accepted, fb_written = await self._try_accept(
                    spec, it, last_answer, hist
                )
                if accepted:
                    self._record(hist, it)
                    stop_cause = "verified"
                    break
                wrote_fb = wrote_fb or fb_written

            # A normal (non-error) iteration completed: reset the
            # consecutive-error tracker so "same error twice" means
            # truly back-to-back.
            last_err_sig = None

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
            if self._plateaued(hist):
                stop_cause = "stagnation"
                break
            # Progress-aware extension: a slow but converging model is not
            # capped at max_iterations while it is still advancing - bounded
            # by max_extra_iterations and the inviolable token ceiling.
            if (
                n >= effective_max
                and self._advancing(hist)
                and total_tokens <= self._cfg.max_total_tokens
                and (effective_max - self._cfg.max_iterations)
                < self._cfg.max_extra_iterations
            ):
                effective_max += 1

        # Gap #4: climb the escalation ladder before giving up. When the local
        # loop is stuck (tried and could not pass validation) and an escalation
        # path was injected, run ONE attempt on the stronger model and re-judge.
        if (
            stop_cause in _STUCK_STOP_CAUSES
            and self._escalate is not None
        ):
            self._append_progress("ESCALATE one attempt on a stronger model (stuck)")
            accepted, last_answer = await self._escalated_attempt(
                spec, hist, last_answer, len(hist) + 1
            )
            if accepted:
                stop_cause = "verified"

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

    async def _escalated_attempt(
        self,
        spec: GoalSpec,
        hist: list[GoalIteration],
        last_answer: str,
        n: int,
    ) -> tuple[bool, str]:
        """Run one fresh attempt on the stronger model and judge it with the same
        validation/acceptance path as a normal iteration. Returns
        ``(accepted, last_answer)``. Never raises; a failed escalation just leaves
        the original stuck outcome in place."""
        if self._escalate is None:
            return False, last_answer
        try:
            coro = self._escalate(self._build_prompt(spec, n), n)
            trace = (
                await asyncio.wait_for(coro, timeout=self._cfg.iteration_timeout_s)
                if self._cfg.iteration_timeout_s > 0
                else await coro
            )
        except Exception as exc:
            log.warning("goal_loop_escalate_error", error=str(exc)[:200])
            return False, last_answer

        ans = self._answer_of(trace) or last_answer
        it = GoalIteration(
            n=n,
            reason=trace.reason or "",
            evidence=float(getattr(trace, "evidence_score", 0.0) or 0.0),
            tokens=int(getattr(trace, "total_tokens_used", 0) or 0),
            verdict=self._verdict(trace),
        )
        if it.verdict == "verified":
            accepted, _ = await self._try_accept(spec, it, ans, hist)
            if accepted:
                self._record(hist, it)
                return True, ans
        self._record(hist, it)
        return False, ans

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

    async def _try_accept(
        self,
        spec: GoalSpec,
        it: GoalIteration,
        last_answer: str,
        hist: list[GoalIteration],
    ) -> tuple[bool, bool]:
        """Run the loop's own deterministic validation (the real arbiter)
        and, on pass, the adversarial review. Returns
        ``(accepted, feedback_written)``. Shared by the inner-"completed"
        path and the C2 decoupled path, so a budget-exhausted / inner-gated
        iteration whose on-disk work already passes is not discarded."""
        passed, detail = await self._run_validation(spec)
        it.validation_passed = passed
        if passed:
            allow, why = await self._adversarial_review(spec, last_answer, detail)
            it.review_passed = allow
            if allow:
                return True, False
            it.verdict = "progress"
            self._append_progress(
                f"Avoid: adversarial review blocked: {why[:200]}"
            )
            self._write_feedback("adversarial-review-block", why)
            return False, True
        it.verdict = "progress"
        self._append_progress(f"Avoid: validation failed: {detail[:200]}")
        self._write_feedback("validation-failed", detail)
        return False, True

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
