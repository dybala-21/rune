"""Check artifact outcomes before completion when RUNE_BENCH_EVIDENCE_GATE is enabled."""

from __future__ import annotations

import json

from rune.utils.env import env_flag, env_int
from rune.utils.logger import get_logger

log = get_logger(__name__)

_EVIDENCE_GATE_ENV = "RUNE_BENCH_EVIDENCE_GATE"
_EVIDENCE_GATE_TIMEOUT_MS_ENV = "RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS"
# Bound finalization latency; a timeout is inconclusive, never a passing check.
_DEFAULT_CHECK_TIMEOUT_MS = 30_000
_MAX_EVIDENCE_OUTPUT_CHARS = 4_000

_EXTRACT_SYSTEM = (
    "You translate a coding-benchmark task into ONE self-contained POSIX shell "
    "script that verifies the produced artifact against the task's own success "
    "criteria, using only public task files (never hidden evaluator paths such "
    "as /tests or /oracle).\n"
    "Rules for the script you emit:\n"
    "- Exit 0 only if EVERY stated success criterion holds; exit non-zero otherwise.\n"
    "- On failure, print the FIRST concrete mismatch (e.g. the first differing "
    "line/byte with got vs expected, or which constraint was violated).\n"
    "- Re-run the task's required entrypoint the way the task describes "
    "(same command/flags); do not assume the artifact already ran.\n"
    "- Operate on a COPY of any input you mutate so the real input file is "
    "preserved; clean up temp files you create.\n"
    "- THE CHECK MUST FINISH IN A FEW SECONDS. Do NOT run the artifact over a "
    "large input in full. If the input has more than ~1000 rows/lines, you MUST "
    "verify on MULTIPLE DISJOINT SAMPLES, not just the first rows — a first-rows-"
    "only check is easily passed by an artifact that fails elsewhere (sampling "
    "blind spot). Build a sample that splices together the FIRST ~100 rows, a "
    "MIDDLE ~100 rows, and the LAST ~100 rows of the public input, build the "
    "matching expected sample from the SAME line ranges of the public expected "
    "output (preserving order), run the artifact on that spliced sample copy, and "
    "compare. (e.g. with awk/sed select line ranges 1-100, mid-100, last-100 from "
    "both $INPUT and $EXPECTED into $tmpdir/in and $tmpdir/exp, run the artifact "
    "on $tmpdir/in, then `cmp -s`.) This only works when the transform is "
    "per-row/independent (so splicing preserves correctness). Process the FULL "
    "input instead ONLY if correctness depends on cross-row context (sorting, "
    "dedup, totals, reordering across lines). Never copy a multi-hundred-thousand-"
    "row file just to re-run the transform.\n"
    "- Use only commands available in a minimal container (sh, cmp, diff, head, "
    "sed, awk, the task's own required tools). No network.\n"
    "- POSIX only — the check also runs on macOS, where `timeout`, `head -n-N`, "
    "`tail -n+N` and `grep -P` do NOT exist. Never use them.\n"
    "- If the task is a SERVICE, do not start a server and curl a fixed port: "
    "ports collide across parallel checks and a leaked server fails every later "
    "one. Exercise it through the project's own in-process entry point instead "
    "(its test runner, or a short program that calls the handler directly). Only "
    "bind a port if there is no other way, and then bind port 0 and read back "
    "the assigned port rather than hardcoding one.\n"
    "- If the task's success criteria are not mechanically checkable from public "
    "files, output exactly the token NO_CHECK and nothing else.\n"
    "Output ONLY the script body (or NO_CHECK). No markdown fences, no prose."
)


def evidence_gate_enabled() -> bool:
    return env_flag(_EVIDENCE_GATE_ENV)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # drop first fence line and any trailing fence
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


async def extract_success_check(instruction: str) -> str | None:
    """Generate a shell check, or return None when no mechanical check can be produced."""
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        # Use the best tier to generate the artifact check.
        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"Task:\n{instruction}\n\nVerification script:"},
            ],
            tier=ModelTier.BEST,
            max_tokens=900,
            timeout=30.0,
        )
    except Exception as exc:  # pragma: no cover - network/SDK variance
        log.warning("evidence_gate_extract_failed", error=str(exc)[:120])
        return None

    text = ""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "") or ""
    else:
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""

    script = _strip_fences(text)
    if not script or "NO_CHECK" in script:
        log.info("evidence_gate_no_check")
        return None
    return script


async def run_evidence_check(script: str, cwd: str) -> tuple[str, str]:
    """Check in the run's environment; missing verdicts remain inconclusive."""
    from rune.safety.verification import run_check

    timeout_ms = env_int(_EVIDENCE_GATE_TIMEOUT_MS_ENV, _DEFAULT_CHECK_TIMEOUT_MS)
    result = await run_check(script, cwd, max(1.0, timeout_ms / 1000.0))
    output = (result.stdout.decode("utf-8", "replace") + result.error)[:_MAX_EVIDENCE_OUTPUT_CHARS]
    if result.code is None:
        return "skip", output
    return ("pass" if result.code == 0 else "fail"), output


def build_block_message(evidence: str) -> str:
    return (
        "[Evidence Gate] Your own success-criteria check on the produced artifact "
        "FAILED. Do not finalize. Fix the artifact so the check passes. First "
        "mismatch / violated constraint:\n"
        + (evidence or "(no output captured)")
    )


class EvidenceGate:
    """Extract a check once per run and re-execute it on completion attempts."""

    __slots__ = (
        "_instruction",
        "_cwd",
        "_script",
        "_extracted",
        "_spec",
        "_spec_extracted",
        "_test_snapshot",
        "_guard_note",
        "verdict_counts",
        "last_verdict",
        "last_evidence",
        "last_tests_restored",
    )

    def __init__(self, instruction: str, cwd: str) -> None:
        self._instruction = instruction
        self._cwd = cwd
        # Snapshot existing tests so later agent edits cannot weaken verification.
        from rune.agent.validation_guard import snapshot_tests

        self._test_snapshot = snapshot_tests(cwd)
        self.last_tests_restored: list[str] = []
        self._guard_note = ""
        self._script: str | None = None
        self._extracted = False
        # Prefer a structured spec; fall back to a generated script if extraction fails.
        self._spec: object | None = None
        self._spec_extracted = False
        # Persist decisions in CompletionTrace when benchmark logs are unavailable.
        self.verdict_counts: dict[str, int] = {"pass": 0, "fail": 0, "skip": 0}
        self.last_verdict: str = ""
        self.last_evidence: str = ""

    async def verdict(self) -> tuple[str, str | None]:
        """Return (pass/fail/skip, message); only a failed check blocks completion."""
        # Restore modified original tests before verification and disclose the restoration.
        from rune.agent.validation_guard import restoration_note, restore_tests

        report = restore_tests(self._test_snapshot)
        self.last_tests_restored = report.restored + report.quarantined
        self._guard_note = restoration_note(report)

        # Preferred: spec-driven verification (deterministic sampling/run/compare).
        if not self._spec_extracted:
            from rune.agent.evidence_spec import extract_spec

            self._spec = await extract_spec(self._instruction)
            self._spec_extracted = True
        if self._spec is not None:
            return await self._verdict_from_spec()

        # Fallback: legacy LLM-emitted script path.
        if not self._extracted:
            self._script = await extract_success_check(self._instruction)
            self._extracted = True
        if self._script is None:
            return self._record("skip", None, "")
        state, evidence = await run_evidence_check(self._script, self._cwd)
        if state == "pass":
            return self._record("pass", None, "")
        if state == "skip":
            # A timeout or spawn error provides no evidence of success.
            return self._record("skip", None, "")
        return self._record("fail", build_block_message(evidence), evidence)

    async def _verdict_from_spec(self) -> tuple[str, str | None]:
        """Recheck the full file; an inconclusive result retains the sample pass."""
        from rune.agent.evidence_spec import VerificationSpec, run_spec

        spec = self._spec
        assert isinstance(spec, VerificationSpec)
        state, evidence = await run_spec(spec, full_file=False)
        if state == "fail":
            return self._record("fail", build_block_message(evidence), evidence)
        if state == "skip":
            return self._record("skip", None, "")
        # Confirm against the full file; an inconclusive result retains the sample pass.
        full_state, full_evidence = await run_spec(spec, full_file=True)
        if full_state == "fail":
            return self._record("fail", build_block_message(full_evidence), full_evidence)
        return self._record("pass", None, "")

    def _record(
        self, state: str, message: str | None, evidence: str
    ) -> tuple[str, str | None]:
        note = self._guard_note
        if note:
            evidence = f"{note}\n{evidence}" if evidence else note
            if message:
                message = f"{note}\n{message}"
        self.verdict_counts[state] = self.verdict_counts.get(state, 0) + 1
        self.last_verdict = state
        self.last_evidence = evidence[:500]
        return state, message

    def summary(self) -> dict[str, object]:
        """Persistable decision history (surfaced via CompletionTrace)."""
        return {
            "mode": "spec" if self._spec is not None else "script",
            "extracted": self._extracted or self._spec_extracted,
            "has_check": self._spec is not None or self._script is not None,
            "verdict_counts": dict(self.verdict_counts),
            "last_verdict": self.last_verdict,
            "last_evidence": self.last_evidence[:200],
            "tests_restored": list(self.last_tests_restored),
        }

    async def check(self) -> str | None:
        """Return the failure message, or None for a pass or inconclusive check."""
        _state, message = await self.verdict()
        return message

    def describe(self) -> str:
        """Compact JSON describing gate state (for audit/debug)."""
        return json.dumps(
            {"extracted": self._extracted, "has_check": self._script is not None}
        )
