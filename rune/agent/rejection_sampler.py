"""Verifier-guided rejection sampling (best-of-K).

Run K *independent* fresh-context attempts at a goal and keep the first one a
verifier accepts. This turns model nondeterminism into a selection signal: if a
single attempt passes with probability p, then K attempts + select succeeds with
probability 1-(1-p)^K, which rises fast even for small p.

This is the counterpart to inline self-fix (``auto_verify``): when a model is too
weak to repair its own output from an injected error, re-running it fresh and
*selecting* a good sample works where in-place repair does not. Empirically (this
repo's gemini calc bench) inline self-fix gave 0/5 while plain sampling gave 1/5,
so the verifier-as-selector path is the one that scales down to weak models.

The sampler is execution-agnostic: callers supply ``run_attempt`` (produce a
candidate) and ``verify`` (accept/reject it). Attempts must be independent —
each a fresh context — so failures don't correlate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rune.utils.env import env_flag
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Verify with a test written in the project's own framework before falling back
# to the Evidence Gate. Default OFF: a generated test that is too weak would
# PASS a wrong candidate, which is worse than the missed selection it replaces,
# so it stays opt-in until an A/B measures its false-positive rate.
_GENERATED_TEST_ENV = "RUNE_GENERATED_TEST_VERIFY"

# Targeted tests: run the repo's own test files nearest to the changed
# sources — cheaper than the full suite, stronger than a synthetic check.
# Only seed-canonical test files count, restored before running.
_TARGETED_TEST_TIMEOUT_S = 120.0
_TARGETED_TEST_MAX_FILES = 5
_TARGETED_CHANGED_CAP = 50  # give up mapping if the candidate rewrote the world


def _changed_source_files(cwd: str, seed_cwd: str) -> list[str]:
    """Relpaths of files in *cwd* that are new or modified vs the seed tree.

    Seeding copies with mtime preserved (copy2), so an edited file differs from
    the seed original in mtime or size. Capped: a whole-tree rewrite means the
    diff signal is broken, and targeted mapping would be meaningless.
    """
    import os

    changed: list[str] = []
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(cwd):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, cwd)
            try:
                st = os.stat(full)
            except OSError:
                continue
            try:
                seed_st = os.stat(os.path.join(seed_cwd, rel))
            except OSError:
                changed.append(rel)  # new file
            else:
                if (seed_st.st_mtime, seed_st.st_size) != (st.st_mtime, st.st_size):
                    changed.append(rel)
            if len(changed) > _TARGETED_CHANGED_CAP:
                return []
    return sorted(changed)


_TEST_ENUM_CAP = 3000  # test files scanned per repo
_TEST_READ_CAP = 200_000  # bytes read per test file for import grep


def _enumerate_test_files(root: str) -> list[str]:
    """Conventional test files: test_*.py, *_test.py, tests.py, under tests/."""
    import os

    out: list[str] = []
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox",
            "build", "dist", ".mypy_cache", ".pytest_cache"}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        rel_dir = os.path.relpath(dirpath, root)
        in_tests = "tests" in rel_dir.replace("\\", "/").split("/")
        for fn in files:
            if not fn.endswith(".py") or fn in ("__init__.py", "conftest.py"):
                continue
            if (fn.startswith("test_") or fn.endswith("_test.py")
                    or fn == "tests.py" or in_tests):
                out.append(os.path.normpath(os.path.join(rel_dir, fn)))
                if len(out) >= _TEST_ENUM_CAP:
                    return out
    return out


def _module_dotted(rel: str) -> tuple[str, str]:
    """('a.b.mod', 'mod') for a/b/mod.py; packages use their dir path."""
    parts = rel[:-3].replace("\\", "/").split("/")
    if parts and parts[0] == "src":  # src layout
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts), (parts[-1] if parts else "")


def _stem_token(t: str) -> str:
    """Light plural stemming: queries->query, indexes->index, models->model."""
    t = t.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 3:
        return t[:-2]
    if t.endswith("s") and len(t) > 2:
        return t[:-1]
    return t


def _tokens_similar(a: str, b: str) -> bool:
    """query ~ queries etc. — plural stemming + close similarity."""
    import difflib

    a, b = _stem_token(a), _stem_token(b)
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    return difflib.SequenceMatcher(a=a, b=b).ratio() >= 0.85


def _targeted_test_files(cwd: str, seed_cwd: str) -> list[str]:
    """Seed-canonical test files covering the candidate's changed sources.

    Layered static mapper (no LLM, runs in seconds):
      L1 reverse import grep — test files importing the changed module (or
         importing its stem from the parent package); precision anchor.
      L2 name/path conventions — test_<stem>.py near the module;
         path-component join (sphinx/ext/autodoc -> test_ext_autodoc.py);
         django-style tests/<topic>/ directory-token match.
    Only files present in the SEED count (agent-written tests are not
    independent evidence). Ranked L1 before L2, capped.
    """
    import os
    import re

    changed = [
        rel for rel in _changed_source_files(cwd, seed_cwd)
        if rel.endswith(".py")
        and not os.path.basename(rel).startswith("test")
    ]
    if not changed:
        return []
    test_files = _enumerate_test_files(seed_cwd)
    if not test_files:
        return []

    l1: list[str] = []
    l2: list[str] = []

    for rel in changed:
        dotted, stem = _module_dotted(rel)
        if not stem:
            continue
        parts = dotted.split(".")
        parent = ".".join(parts[:-1])

        # --- L1: reverse import grep over the seed's test files
        pats = [
            re.compile(rf"^\s*from\s+{re.escape(dotted)}\s+import", re.M),
            re.compile(rf"^\s*import\s+{re.escape(dotted)}\b", re.M),
        ]
        if parent:
            pats.append(re.compile(
                rf"^\s*from\s+{re.escape(parent)}\s+import\s+[^\n]*"
                rf"\b{re.escape(stem)}\b", re.M,
            ))
        for tf in test_files:
            if tf in l1:
                continue
            try:
                with open(os.path.join(seed_cwd, tf), "rb") as fh:
                    src = fh.read(_TEST_READ_CAP).decode("utf-8", "replace")
            except OSError:
                continue
            if any(p.search(src) for p in pats):
                l1.append(tf)

        # --- L2a: stem convention near the module
        d = os.path.dirname(rel)
        for cand in (
            os.path.join(d, f"test_{stem}.py"),
            os.path.join(d, "tests", f"test_{stem}.py"),
            os.path.join("tests", f"test_{stem}.py"),
        ):
            cand = os.path.normpath(cand)
            if cand not in l2 and os.path.isfile(os.path.join(seed_cwd, cand)):
                l2.append(cand)
        # --- L2b: path-component join (test_ext_autodoc.py style) — both
        # with the module stem and package-only (a module inside a package
        # frequently maps to a package-level test file).
        joins = set()
        comps = parts[1:] if len(parts) > 1 else parts
        pkg_comps = comps[:-1]  # without the module stem
        for seq in (comps, pkg_comps):
            for i in range(len(seq)):
                joins.add("test_" + "_".join(seq[i:]) + ".py")
        for tf in test_files:
            if tf not in l2 and os.path.basename(tf) in joins:
                l2.append(tf)
        # --- L2c: directory-token match (django tests/<topic>/tests.py)
        mod_tokens = {t for t in parts if t not in ("src",)}
        for tf in test_files:
            if tf in l2 or tf in l1:
                continue
            tf_dirs = os.path.dirname(tf).replace("\\", "/").split("/")
            topic = [t for t in tf_dirs if t not in ("tests", "test", "")]
            if topic and any(
                _tokens_similar(t, m) for t in topic for m in mod_tokens
            ):
                l2.append(tf)

    # L1 outranks L2. If L1 exploded (django: hundreds import the package),
    # keep the ones whose grep hit the FULL dotted path first via order and cap.
    ordered = l1 + [t for t in l2 if t not in l1]
    return ordered[:_TARGETED_TEST_MAX_FILES]


def _project_python(seed_cwd: str) -> str:
    """Project venv python if present, else RUNE's own interpreter.

    Old repos often can't even import under RUNE's interpreter, which would
    leave the verifier with no execution signal.
    """
    import os
    import sys

    for name in (".venv", "venv"):
        cand = os.path.join(seed_cwd, name, "bin", "python")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return sys.executable


def _restore_canonical_tests(cwd: str, seed_cwd: str, tests: list[str]) -> None:
    """Overwrite the candidate's copies of *tests* with the seed originals.

    The tests are the yardstick; a candidate that edited them must not be
    graded against its own edits (same rationale as validation_guard).
    """
    import os
    import shutil

    for rel in tests:
        try:
            dst = os.path.join(cwd, rel)
            os.makedirs(os.path.dirname(dst) or cwd, exist_ok=True)
            shutil.copy2(os.path.join(seed_cwd, rel), dst)
        except OSError as exc:
            log.warning("targeted_test_restore_failed", path=rel,
                        error=str(exc)[:120])


@dataclass
class Attempt[T]:
    """One sampled candidate and the verifier's verdict on it."""

    index: int
    candidate: T
    passed: bool


@dataclass
class RejectionResult[T]:
    """Outcome of a best-of-K run."""

    selected: T | None  # first candidate the verifier accepted, else None
    selected_index: int | None
    attempts: list[Attempt[T]]

    @property
    def solved(self) -> bool:
        return self.selected is not None

    @property
    def pass_count(self) -> int:
        return sum(1 for a in self.attempts if a.passed)


async def solve_with_rejection[T](
    run_attempt: Callable[[int], Awaitable[T]],
    verify: Callable[[T], Awaitable[bool]],
    k: int,
    *,
    stop_on_first_pass: bool = True,
) -> RejectionResult[T]:
    """Sample up to ``k`` independent candidates; the verifier selects.

    ``run_attempt(i)`` produces candidate i (must be a fresh, independent run).
    ``verify(candidate)`` returns whether it is acceptable. By default we stop at
    the first accepted candidate (cheapest path to a solution); set
    ``stop_on_first_pass=False`` to sample all k (e.g. to measure pass rate).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    attempts: list[Attempt[T]] = []
    selected: T | None = None
    selected_index: int | None = None

    for i in range(k):
        candidate = await run_attempt(i)
        passed = await verify(candidate)
        attempts.append(Attempt(index=i, candidate=candidate, passed=passed))
        log.info("rejection_attempt", index=i, passed=passed)
        if passed and selected is None:
            selected, selected_index = candidate, i
            if stop_on_first_pass:
                break

    log.info(
        "rejection_result",
        k=k,
        solved=selected is not None,
        selected_index=selected_index,
        sampled=len(attempts),
    )
    return RejectionResult(
        selected=selected, selected_index=selected_index, attempts=attempts
    )


async def sample_parallel[T](
    run_attempt: Callable[[int], Awaitable[T]],
    verify: Callable[[T], Awaitable[bool]],
    k: int,
) -> RejectionResult[T]:
    """Like ``solve_with_rejection`` but run all ``k`` attempts concurrently.

    Faster wall-clock when attempts are independent and I/O-bound (e.g. each is a
    subprocess LLM call). Always samples all k (no early stop); the verifier then
    selects the lowest-index accepted candidate.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    candidates = await asyncio.gather(*(run_attempt(i) for i in range(k)))
    verdicts = await asyncio.gather(*(verify(c) for c in candidates))
    attempts = [
        Attempt(index=i, candidate=c, passed=v)
        for i, (c, v) in enumerate(zip(candidates, verdicts, strict=True))
    ]
    chosen = next((a for a in attempts if a.passed), None)
    log.info(
        "rejection_result_parallel",
        k=k,
        solved=chosen is not None,
        selected_index=chosen.index if chosen else None,
    )
    return RejectionResult(
        selected=chosen.candidate if chosen else None,
        selected_index=chosen.index if chosen else None,
        attempts=attempts,
    )


async def make_evidence_gate_verifier(
    instruction: str,
) -> Callable[[str], Awaitable[bool]]:
    """Build a ``verify(cwd)`` that uses RUNE's Evidence Gate as the selector.

    The success check is extracted ONCE (it depends only on the instruction, not
    the candidate), then each call re-runs it against a candidate's working dir.
    Only an actual ``"pass"`` selects a candidate — ``"fail"``/``"skip"``/no-check
    all return False (conservative: never select an unverified candidate).

    Measured on a hard arithmetic task: FP=0 (never passes a wrong solution) and
    ~87% correct-pass, which makes it a SAFE selector for best-of-K — a wrong pick
    is far costlier than missing one good candidate, and a larger K covers the
    occasional false-negative.
    """
    from rune.agent.evidence_gate import extract_success_check, run_evidence_check

    script = await extract_success_check(instruction)
    if not script:
        # No mechanical check available: the verifier will reject every
        # candidate. Log once here so a best-of-K that selects nothing is
        # explainable (rather than silently failing).
        log.info("rejection_eg_verifier_no_check")

    # Keep each failed candidate's mismatch evidence (keyed by cwd) so callers
    # can learn a correctness rule from it (best-of failure-driven learning).
    evidence_by_cwd: dict[str, str] = {}

    async def verify(cwd: str) -> bool:
        if not script:
            return False
        state, evidence = await run_evidence_check(script, cwd)
        if state == "fail" and evidence:
            evidence_by_cwd[cwd] = evidence
        return state == "pass"

    # Expose whether a mechanical check exists so callers can distinguish
    # "checked but every candidate failed" from "no check could be built, so
    # best-of-K structurally cannot select anything" — two very different
    # outcomes that otherwise both look like 0/K passed.
    verify.has_check = bool(script)  # type: ignore[attr-defined]
    verify.evidence_by_cwd = evidence_by_cwd  # type: ignore[attr-defined]
    return verify


async def make_verifier(
    instruction: str, seed_cwd: str | None = None
) -> Callable[[str], Awaitable[bool]]:
    """Build a best-of-K ``verify(cwd)`` that PREFERS execution over LLM-judge.

    For code, running the project's tests is a more reliable selector than an
    LLM/Evidence-Gate judge, and the gap is largest for weak models (arXiv
    2502.14382 / 2506.10056) — which is exactly the best-of-K regime. So each
    candidate is verified by:

    1. its repo test command, if one is detectable (``detect_test_command``) —
       ``pass`` selects, ``fail`` rejects (keeping the output as evidence);
    2. otherwise (or when tests are inconclusive/``skip``) the Evidence Gate.

    Greenfield tasks with no tests fall back to the Evidence Gate exactly as
    before, so this never regresses the no-tests case; it strictly adds an
    execution path for tested repos (e.g. ``--include-cwd``).
    """
    from rune.agent.auto_verify import detect_test_command, run_verify

    eg = await make_evidence_gate_verifier(instruction)
    evidence_by_cwd: dict[str, str] = getattr(eg, "evidence_by_cwd", {})
    # A check exists if the Evidence Gate built one OR the seeded tree has tests.
    has_check = bool(getattr(eg, "has_check", False)) or (
        bool(seed_cwd) and detect_test_command(seed_cwd) is not None
    )

    # Which check decided each candidate's verdict (keyed by cwd), so callers
    # can report what the winner passed (test command vs Evidence Gate).
    method_by_cwd: dict[str, str] = {}

    # The full suite's inconclusive verdict (timeout/collection error) holds
    # for every sibling candidate — pay it once.
    _suite_unusable = False

    # Contract-derived test, built at most once per task and reused unchanged so
    # every candidate faces the same bar. The "done" key separates "not built
    # yet" from "built, unusable".
    _gen_cache: dict[str, object] = {}
    _gen_lock = asyncio.Lock()

    async def _generated_test_verdict(cwd: str) -> tuple[str, str]:
        """Run the contract-derived test against *cwd*: (state, evidence)."""
        if not env_flag(_GENERATED_TEST_ENV):
            return "skip", ""
        from rune.agent.generated_test import (
            detect_framework,
            discriminates,
            extract_public_api,
            generate_verification_test,
            run_generated_test,
        )

        async with _gen_lock:
            if "done" not in _gen_cache:
                _gen_cache["done"] = True
                # The pre-edit tree: safe to read API signatures from, and the
                # yardstick the test must not be able to pass.
                baseline = seed_cwd
                fw = detect_framework(baseline or cwd)
                if fw:
                    api = extract_public_api(baseline or cwd, fw)
                    body = await generate_verification_test(instruction, fw, api)
                    if body and baseline and not await discriminates(
                        body, fw, baseline
                    ):
                        body = None
                    if body:
                        _gen_cache["fw"] = fw
                        _gen_cache["body"] = body
                        log.info("generated_test_ready", framework=fw.name)
        fw = _gen_cache.get("fw")
        body = _gen_cache.get("body")
        if not fw or not body:
            return "skip", ""
        return await run_generated_test(body, fw, cwd)  # type: ignore[arg-type]

    async def verify(cwd: str) -> bool:
        nonlocal _suite_unusable
        cmd = None if _suite_unusable else detect_test_command(cwd)
        if cmd:
            # Use the project interpreter and shadow with the candidate
            # tree, same as targeted tests below.
            import re as _re
            import sys as _sys
            if seed_cwd and cmd[0] == _sys.executable:
                cmd = [
                    "/usr/bin/env", f"PYTHONPATH={cwd}:{cwd}/src",
                    _project_python(seed_cwd), *cmd[1:],
                ]
            state, evidence = await run_verify(cmd, cwd)
            if state == "fail" and not _re.search(r"\b\d+ failed\b", evidence):
                # Non-zero exit without failed tests = collection/usage error
                # (e.g. interpreter mismatch, missing plugin) — inconclusive,
                # not a rejection; fall through to the targeted-test path.
                log.info("verify_suite_inconclusive", cmd=" ".join(cmd[:4]))
                state, evidence = "skip", ""
                _suite_unusable = True  # don't re-pay this per sibling
            elif state == "skip":
                _suite_unusable = True  # spawn error / timeout: same next time
            if state in ("pass", "fail"):
                method_by_cwd[cwd] = f"`{' '.join(cmd)}`"
            if state == "pass":
                # Report the test count: passing the provided tests is not proof
                # of correctness, so the count lets the user gauge the check.
                from rune.agent.auto_verify import assertions_ran, passed_test_count
                n = passed_test_count(evidence)
                if n is not None:
                    method_by_cwd[cwd] = (
                        f"`{' '.join(cmd)}`, {n} test{'s' if n != 1 else ''}"
                    )
                # An exit-0 run that asserted nothing is not evidence: it passes
                # every candidate, so best-of-K collapses to "pick #0" while
                # reporting it verified. Fall through to a real check instead.
                # Unparseable summaries stay trusted — never block work over a
                # runner we don't recognize.
                if assertions_ran(evidence) is not False:
                    return True
                log.info("verify_suite_vacuous", cmd=" ".join(cmd))
            elif state == "fail":
                if evidence:
                    evidence_by_cwd[cwd] = evidence
                return False
            # "skip" (could not run): fall through too.

        # Targeted repo tests: the seed's own test files nearest to the changed
        # sources. Real held-out tests beat any synthetic check, and running
        # only the relevant files fits the timeout that the full suite blows.
        if seed_cwd:
            targets = _targeted_test_files(cwd, seed_cwd)
            if targets:
                import re as _re

                _restore_canonical_tests(cwd, seed_cwd, targets)
                # Shadow the venv's editable install with the candidate
                # tree, or the seed's unfixed code gets graded instead.
                t_state, t_evidence = await run_verify(
                    [
                        "/usr/bin/env",
                        f"PYTHONPATH={cwd}:{cwd}/src",
                        _project_python(seed_cwd),
                        "-m", "pytest", "-q", *targets,
                    ],
                    cwd,
                    timeout=_TARGETED_TEST_TIMEOUT_S,
                )
                # pytest exits non-zero for collection/usage errors too; only a
                # run whose summary reports failed tests is a real rejection —
                # anything else (import error under a mismatched interpreter,
                # bad usage) is inconclusive, not evidence against the fix.
                if t_state == "fail" and not _re.search(
                    r"\b\d+ failed\b", t_evidence
                ):
                    t_state = "skip"
                if t_state == "pass":
                    from rune.agent.auto_verify import assertions_ran
                    if assertions_ran(t_evidence) is not False:
                        method_by_cwd[cwd] = (
                            f"targeted tests ({', '.join(targets)})"
                        )
                        # Provisional: pre-fix code passes these tests too,
                        # so a pass selects but never verifies.
                        verify.provisional_by_cwd[cwd] = True  # type: ignore[attr-defined]
                        log.info("targeted_tests_pass", files=targets)
                        return True
                elif t_state == "fail":
                    method_by_cwd[cwd] = "targeted tests"
                    evidence_by_cwd[cwd] = t_evidence
                    log.info("targeted_tests_fail", files=targets)
                    return False
                log.info("targeted_tests_inconclusive", files=targets)

        # Tried before the Evidence Gate: a test in the project's own framework
        # runs in-process, which the Gate's generated shell script cannot do —
        # it can only reach a service through a real port, and those collide
        # across candidates.
        gen_state, gen_evidence = await _generated_test_verdict(cwd)
        if gen_state == "pass":
            method_by_cwd[cwd] = "generated contract test"
            return True
        if gen_state == "fail":
            method_by_cwd[cwd] = "generated contract test"
            if gen_evidence:
                evidence_by_cwd[cwd] = gen_evidence
            return False

        method_by_cwd[cwd] = "Evidence Gate"
        # Set when the EG check passed the unfixed baseline: such a check
        # accepts anything, so its pass never counts as verified.
        if getattr(verify, "eg_disabled", False):
            return False
        return await eg(cwd)

    verify.eg_disabled = False  # type: ignore[attr-defined]
    verify.has_check = has_check  # type: ignore[attr-defined]
    verify.provisional_by_cwd = {}  # type: ignore[attr-defined]
    verify.evidence_by_cwd = evidence_by_cwd  # type: ignore[attr-defined]
    verify.method_by_cwd = method_by_cwd  # type: ignore[attr-defined]
    return verify
