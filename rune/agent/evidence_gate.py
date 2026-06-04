"""Evidence Gate — deterministic output-correctness verification.

RUNE's Completion Gate (``completion_gate.py``) verifies *behavioral* evidence
(did the agent read/write/execute?) but not *outcome* evidence (is the produced
artifact actually correct?). That gap let benchmark attempts finalize with a
byte-wrong or rule-violating artifact (see the v8 ``large-scale-text-editing``
canaries: a macro that ran but used a banned construct, or transformed to the
wrong bytes, still scored "verified").

This module fills that gap for benchmark runs. It does NOT hardcode any
task-specific rule (that would violate the no-pattern-matching principle).
Instead it asks the model to translate the task's own success criteria into a
single self-contained shell check, runs that check before finalization, and
blocks completion (returning first-mismatch evidence) when it fails.

Off by default; enabled via ``RUNE_BENCH_EVIDENCE_GATE``. Conservative by
construction: if a check cannot be produced or cannot run, it does NOT block
(never converts a real success into a false failure).
"""

from __future__ import annotations

import json

from rune.utils.env import env_flag, env_int
from rune.utils.logger import get_logger

log = get_logger(__name__)

_EVIDENCE_GATE_ENV = "RUNE_BENCH_EVIDENCE_GATE"
_EVIDENCE_GATE_TIMEOUT_MS_ENV = "RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS"
# Sample-based checks should finish in seconds; a short default keeps a slow or
# accidentally full-file check from stalling finalize. On timeout the verdict is
# "skip" (inconclusive), never "pass", so a too-short timeout cannot fabricate a
# completion — it only forgoes the gate's help for that attempt.
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
    """Ask the model for a shell script that verifies the task's success criteria.

    Returns the script body, or ``None`` when no mechanical check is possible
    (the model answers NO_CHECK, the call fails, or the output is empty). A
    ``None`` result means "do not block" — the gate stays conservative.
    """
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        # Use the best tier (not the active task model) so the check is at
        # least as reliable as the model that produced the artifact.
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
    """Run the verification script in a direct subprocess.

    Returns ``(state, evidence_text)`` where ``state`` is:
    - ``"pass"``  — script exited 0 (artifact satisfied the check).
    - ``"fail"``  — script exited non-zero (artifact violated the check);
      ``evidence_text`` carries the first-mismatch output.
    - ``"skip"``  — the check could not be run to completion (spawn error or
      TIMEOUT). This is INCONCLUSIVE: it must neither block (don't punish a real
      success we couldn't verify) NOR pass (a slow/timed-out check must NOT be
      read as success — that previously produced false-positive completions
      where a wrong artifact's 1M-row vim check timed out and was treated as
      "pass", finalizing a failing artifact).

    The script runs as a direct subprocess, NOT through the bash capability /
    Guardian. The Evidence Gate is a trusted internal verifier that runs a check
    WE generated to re-verify the agent's artifact; Guardian exists to gate the
    *agent's* actions, and applying it here wrongly blocks safe verifier idioms
    (e.g. `trap 'rm -rf "$tmpdir"' EXIT` on a mktemp dir was blocked as
    "recursive deletion with absolute path", failing the whole check).
    """
    import asyncio

    timeout_ms = env_int(_EVIDENCE_GATE_TIMEOUT_MS_ENV, _DEFAULT_CHECK_TIMEOUT_MS)
    timeout_s = max(1.0, timeout_ms / 1000.0)
    try:
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            script,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # pragma: no cover - spawn failure
        log.warning("evidence_gate_spawn_error", error=str(exc)[:120])
        return "skip", ""

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        log.warning("evidence_gate_timeout", timeout_s=timeout_s)
        return "skip", ""  # inconclusive → neither block nor pass
    except Exception as exc:  # pragma: no cover
        log.warning("evidence_gate_run_error", error=str(exc)[:120])
        return "skip", ""

    output = (stdout.decode("utf-8", errors="replace") if stdout else "")[
        :_MAX_EVIDENCE_OUTPUT_CHARS
    ]
    if proc.returncode == 0:
        log.info("evidence_gate_pass")
        return "pass", output
    log.info("evidence_gate_fail", code=proc.returncode, evidence_len=len(output))
    return "fail", output.strip()


def build_block_message(evidence: str) -> str:
    return (
        "[Evidence Gate] Your own success-criteria check on the produced artifact "
        "FAILED. Do not finalize. Fix the artifact so the check passes. First "
        "mismatch / violated constraint:\n"
        + (evidence or "(no output captured)")
    )


class EvidenceGate:
    """Stateful helper: extract the check once, then verify on demand.

    The extraction is cached for the run (the task instruction does not change),
    so finalization attempts after the first only pay the cheap bash re-run.
    """

    __slots__ = (
        "_instruction",
        "_cwd",
        "_script",
        "_extracted",
        "_spec",
        "_spec_extracted",
        "verdict_counts",
        "last_verdict",
        "last_evidence",
    )

    def __init__(self, instruction: str, cwd: str) -> None:
        self._instruction = instruction
        self._cwd = cwd
        self._script: str | None = None
        self._extracted = False
        # Spec-driven path (preferred): code controls sampling/run/compare; the
        # LLM only supplies the structured spec. Falls back to the legacy script
        # path when no valid spec can be extracted.
        self._spec: object | None = None
        self._spec_extracted = False
        # Observability: structlog is NOT captured in benchmark containers, so
        # the gate records its own decision history for surfacing via the
        # CompletionTrace (which IS persisted to completion_trace.json).
        self.verdict_counts: dict[str, int] = {"pass": 0, "fail": 0, "skip": 0}
        self.last_verdict: str = ""
        self.last_evidence: str = ""

    async def verdict(self) -> tuple[str, str | None]:
        """Run the artifact's success check and return a three-state verdict.

        Returns ``(state, message)`` where ``state`` is:
        - ``"pass"``: a real check ran and the artifact satisfied it. Callers may
          treat this as positive outcome evidence that can OVERRIDE a
          completion-gate "blocked" verdict (behavioral signals are weaker than
          a passing outcome check).
        - ``"fail"``: a real check ran and the artifact failed it; ``message`` is
          a finalize-blocking nudge with first-mismatch evidence.
        - ``"skip"``: no mechanical check could be produced/run; the gate is
          neutral and must NOT influence the decision either way.
        """
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
            # Inconclusive (timeout / spawn error): stay neutral. Do NOT treat a
            # timed-out check as a pass — that finalized wrong artifacts before.
            return self._record("skip", None, "")
        return self._record("fail", build_block_message(evidence), evidence)

    async def _verdict_from_spec(self) -> tuple[str, str | None]:
        """Spec path: multi-disjoint sample check, then a full-file final
        confirmation before accepting a pass (defeats the sampling blind spot)."""
        from rune.agent.evidence_spec import VerificationSpec, run_spec

        spec = self._spec
        assert isinstance(spec, VerificationSpec)
        state, evidence = await run_spec(spec, full_file=False)
        if state == "fail":
            return self._record("fail", build_block_message(evidence), evidence)
        if state == "skip":
            return self._record("skip", None, "")
        # Sample passed — confirm on the FULL file before accepting. If the
        # full check is inconclusive (e.g. times out on huge input), accept the
        # sample pass rather than discarding a likely-correct artifact.
        full_state, full_evidence = await run_spec(spec, full_file=True)
        if full_state == "fail":
            return self._record("fail", build_block_message(full_evidence), full_evidence)
        return self._record("pass", None, "")

    def _record(
        self, state: str, message: str | None, evidence: str
    ) -> tuple[str, str | None]:
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
        }

    async def check(self) -> str | None:
        """Return a block message if the artifact fails its check, else ``None``.

        Thin wrapper over :meth:`verdict` for callers that only care about the
        blocking decision (``"fail"`` → message, ``"pass"``/``"skip"`` → None).
        """
        _state, message = await self.verdict()
        return message

    def describe(self) -> str:
        """Compact JSON describing gate state (for audit/debug)."""
        return json.dumps(
            {"extracted": self._extracted, "has_check": self._script is not None}
        )
