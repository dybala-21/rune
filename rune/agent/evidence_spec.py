"""Spec-driven verification for the Evidence Gate.

Why this exists: letting the LLM emit a whole shell verification *script* proved
unreliable — across runs it randomly produced a full-file check (slow → timeout),
a first-rows-only check (Goodhart blind spot), or a proper multi-sample check.
Prompting could not pin the script's shape (the same run-to-run variance that
dominates this whole problem).

Fix (separation of concerns): the LLM extracts only a small STRUCTURED SPEC
(paths + run-command template + whether the transform is row-independent); the
DETERMINISTIC CODE here does the sampling (first/mid/last disjoint slices),
runs the artifact via a direct subprocess (Guardian-free trusted verifier), and
byte-compares. This removes the LLM's freedom over the parts that caused
timeouts and blind spots, while keeping the LLM only for semantic extraction it
is good at.

External-evidence basis (adversarial design review, 2026-06): a sample verifier
must use MULTIPLE DISJOINT samples, not just first-N (sampling blind spot /
Goodhart, measured 21.8–33% proxy-vs-hidden divergence on SWE-bench); and a
timed-out check must be INCONCLUSIVE ("skip"), never read as pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rune.utils.env import env_int
from rune.utils.logger import get_logger

log = get_logger(__name__)

_SPEC_TIMEOUT_MS_ENV = "RUNE_BENCH_EVIDENCE_GATE_TIMEOUT_MS"
_DEFAULT_CHECK_TIMEOUT_MS = 30_000
_SAMPLE_ROWS_PER_SLICE_ENV = "RUNE_BENCH_EVIDENCE_SAMPLE_ROWS"
_DEFAULT_SAMPLE_ROWS = 100
_LARGE_INPUT_THRESHOLD = 1000  # rows above which we sample instead of full-file
_MAX_EVIDENCE_OUTPUT_CHARS = 4_000

# A run-command template must contain the {INPUT} placeholder so the code can
# substitute the (sampled or full) input copy. Artifact/expected paths are
# substituted too when present.
_INPUT_PLACEHOLDER = "{INPUT}"

_EXTRACT_SPEC_SYSTEM = (
    "You extract a STRUCTURED verification spec (JSON) from a coding-benchmark "
    "task. You do NOT write a shell script. Output ONLY a single JSON object "
    "(no markdown fences, no prose), or the exact token NO_CHECK if the task's "
    "success criteria are not mechanically checkable from public files.\n"
    "JSON fields:\n"
    '  "input_path": absolute path to the public INPUT file the artifact '
    "transforms (e.g. /app/input.csv), or null.\n"
    '  "expected_path": absolute path to the public EXPECTED output to compare '
    "against (e.g. /app/expected.csv), or null.\n"
    '  "artifact_path": absolute path to the artifact the agent must produce '
    "(e.g. /app/apply_macros.vim), or null.\n"
    '  "run_command": the EXACT shell command that applies the artifact to an '
    "input, with the literal token {INPUT} where the input file path goes and "
    "where the command writes its result IN-PLACE to that same {INPUT} copy "
    "(e.g. \"vim -Nu NONE -n -Es {INPUT} -S /app/apply_macros.vim\"). Use the "
    "task's required invocation verbatim. null if there is no such command.\n"
    '  "compare": "in_place" if run_command mutates {INPUT} and the result is '
    'compared to expected; or "stdout" if run_command prints the result to '
    "stdout. Default \"in_place\".\n"
    '  "row_independent": true if the transform is per-row/per-line and '
    "independent across rows (so verifying disjoint row samples is valid); "
    "false if correctness needs cross-row context (sorting, dedup, totals, "
    "reordering across lines).\n"
    "Only use public paths the task names; never reference hidden evaluator "
    "paths (/tests, /oracle). If input_path, expected_path, and run_command "
    "cannot all be determined from the task, output NO_CHECK."
)


@dataclass(frozen=True)
class VerificationSpec:
    input_path: str
    expected_path: str
    artifact_path: str | None
    run_command: str
    compare: str  # "in_place" | "stdout"
    row_independent: bool

    @property
    def valid(self) -> bool:
        return bool(
            self.input_path
            and self.expected_path
            and self.run_command
            and _INPUT_PLACEHOLDER in self.run_command
        )


def _coerce_spec(obj: dict[str, object]) -> VerificationSpec | None:
    try:
        input_path = str(obj.get("input_path") or "").strip()
        expected_path = str(obj.get("expected_path") or "").strip()
        run_command = str(obj.get("run_command") or "").strip()
        artifact_path = obj.get("artifact_path")
        artifact_path = str(artifact_path).strip() if artifact_path else None
        compare = str(obj.get("compare") or "in_place").strip() or "in_place"
        if compare not in ("in_place", "stdout"):
            compare = "in_place"
        row_independent = bool(obj.get("row_independent", False))
    except (AttributeError, TypeError):
        return None
    spec = VerificationSpec(
        input_path=input_path,
        expected_path=expected_path,
        artifact_path=artifact_path,
        run_command=run_command,
        compare=compare,
        row_independent=row_independent,
    )
    return spec if spec.valid else None


def parse_spec(text: str) -> VerificationSpec | None:
    """Parse the LLM's JSON spec output. Returns None on NO_CHECK / invalid."""
    t = (text or "").strip()
    if not t or "NO_CHECK" in t:
        return None
    # Strip accidental markdown fences.
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    # Extract the first JSON object if there's surrounding prose.
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None
        t = m.group(0)
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return _coerce_spec(obj)


async def extract_spec(instruction: str) -> VerificationSpec | None:
    """Ask the model for a structured verification spec (not a script)."""
    try:
        from rune.llm.client import get_llm_client

        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": _EXTRACT_SPEC_SYSTEM},
                {"role": "user", "content": f"Task:\n{instruction}\n\nJSON spec:"},
            ],
            max_tokens=400,
            timeout=30.0,
        )
    except Exception as exc:  # pragma: no cover - network/SDK variance
        log.warning("evidence_spec_extract_failed", error=str(exc)[:120])
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
    spec = parse_spec(text)
    if spec is None:
        log.info("evidence_spec_no_check")
    return spec


# --- deterministic sampling + execution (code-controlled, no LLM freedom) ---


def _read_lines(path: str) -> list[bytes] | None:
    try:
        with open(path, "rb") as f:
            return f.read().splitlines(keepends=True)
    except OSError:
        return None


def build_disjoint_sample(
    input_lines: list[bytes], expected_lines: list[bytes], rows_per_slice: int
) -> tuple[bytes, bytes] | None:
    """Splice first/middle/last row-slices from input and the SAME ranges of
    expected, preserving order. Returns (input_sample, expected_sample) bytes,
    or None when the two files have mismatched line counts (cannot align a
    row-independent sample → caller should fall back to full-file).

    A multi-disjoint sample (not first-rows-only) is what defeats the Goodhart
    blind spot: an artifact wrong only in the middle/end is still caught.
    """
    n = len(input_lines)
    if n != len(expected_lines) or n == 0:
        return None
    if n <= 3 * rows_per_slice:
        # Small enough: just use the whole thing.
        return b"".join(input_lines), b"".join(expected_lines)
    mid_start = (n - rows_per_slice) // 2
    ranges = [
        (0, rows_per_slice),
        (mid_start, mid_start + rows_per_slice),
        (n - rows_per_slice, n),
    ]
    in_parts: list[bytes] = []
    exp_parts: list[bytes] = []
    for lo, hi in ranges:
        in_parts.append(b"".join(input_lines[lo:hi]))
        exp_parts.append(b"".join(expected_lines[lo:hi]))
    return b"".join(in_parts), b"".join(exp_parts)


async def run_spec(spec: VerificationSpec, full_file: bool = False) -> tuple[str, str]:
    """Run the artifact via the spec and byte-compare to expected.

    Returns ``(state, evidence)`` with state in ``"pass"|"fail"|"skip"``.
    - ``full_file=False`` (default): verify on a multi-disjoint row sample when
      the input is large AND row_independent; otherwise full file.
    - ``full_file=True``: always verify on the entire input (final confirmation
      before finalize).
    Timeout / spawn error / unreadable inputs → ``"skip"`` (inconclusive, never
    a false pass).
    """
    import asyncio
    import os
    import tempfile

    input_lines = _read_lines(spec.input_path)
    expected_lines = _read_lines(spec.expected_path)
    if input_lines is None or expected_lines is None:
        return "skip", ""

    # Sample when the input is large and the transform looks row-aligned. We do
    # NOT rely solely on the LLM's row_independent flag (it tends to answer
    # False when unsure, which forces a slow full-file run that then times out).
    # A 1:1 input/expected line count is itself strong evidence of a row-aligned
    # transform; build_disjoint_sample only splices when counts match and
    # returns None otherwise, so this stays safe.
    line_counts_match = len(input_lines) == len(expected_lines)
    use_sample = (
        not full_file
        and (spec.row_independent or line_counts_match)
        and len(input_lines) > _LARGE_INPUT_THRESHOLD
    )
    if use_sample:
        rows = env_int(_SAMPLE_ROWS_PER_SLICE_ENV, _DEFAULT_SAMPLE_ROWS) or _DEFAULT_SAMPLE_ROWS
        sample = build_disjoint_sample(input_lines, expected_lines, rows)
        if sample is None:
            input_bytes = b"".join(input_lines)
            expected_bytes = b"".join(expected_lines)
        else:
            input_bytes, expected_bytes = sample
    else:
        input_bytes = b"".join(input_lines)
        expected_bytes = b"".join(expected_lines)

    timeout_ms = env_int(_SPEC_TIMEOUT_MS_ENV, _DEFAULT_CHECK_TIMEOUT_MS)
    timeout_s = max(1.0, (timeout_ms or _DEFAULT_CHECK_TIMEOUT_MS) / 1000.0)

    tmpdir = tempfile.mkdtemp(prefix="rune-evgate-")
    work = os.path.join(tmpdir, "input_work")
    try:
        with open(work, "wb") as f:
            f.write(input_bytes)
        command = spec.run_command.replace(_INPUT_PLACEHOLDER, work)
        try:
            proc = await asyncio.create_subprocess_exec(
                "sh", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("evidence_spec_spawn_error", error=str(exc)[:120])
            return "skip", ""
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            log.warning("evidence_spec_timeout", timeout_s=timeout_s, full_file=full_file)
            return "skip", ""

        if spec.compare == "stdout":
            produced = stdout or b""
        else:
            try:
                with open(work, "rb") as f:
                    produced = f.read()
            except OSError:
                return "skip", ""

        if proc.returncode != 0:
            head = (stdout or b"")[:_MAX_EVIDENCE_OUTPUT_CHARS].decode("utf-8", "replace")
            log.info("evidence_spec_fail", code=proc.returncode, sampled=use_sample)
            return "fail", f"run_command exited {proc.returncode}\n{head}".strip()

        if produced == expected_bytes:
            log.info("evidence_spec_pass", sampled=use_sample, full_file=full_file)
            return "pass", ""
        # Byte mismatch → report first differing line for a useful nudge.
        evidence = _first_line_mismatch(produced, expected_bytes)
        log.info("evidence_spec_mismatch", sampled=use_sample)
        return "fail", evidence
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def _first_line_mismatch(produced: bytes, expected: bytes) -> str:
    p_lines = produced.splitlines()
    e_lines = expected.splitlines()
    for i in range(max(len(p_lines), len(e_lines))):
        pl = p_lines[i] if i < len(p_lines) else b"<no line>"
        el = e_lines[i] if i < len(e_lines) else b"<no line>"
        if pl != el:
            return (
                f"first mismatch at sampled line {i + 1}:\n"
                f"  expected: {el[:200].decode('utf-8', 'replace')}\n"
                f"  got:      {pl[:200].decode('utf-8', 'replace')}"
            )
    return "outputs differ in length only"
