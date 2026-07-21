"""Verify an artifact with a test written in the project's OWN test framework.

Used when a repo has a test runner with nothing to run, or no runner at all —
cases where the Evidence Gate is the only fallback and a poor one for services.
The Gate verifies by generating a shell script, and a script can only reach a
service through a real socket, so it builds the project, binds a fixed port and
curls it. Measured across five milestones of a Rust/axum project that scored
correct-pass 0/5: best-of-K runs one check per candidate and the servers
collide on (and leak) the port, so every correct implementation was rejected.
The project's own framework does the same job in-process — the equivalent axum
test drives the router directly, no port, ~0.00s.

A test written by the model that wrote the code can just restate that code's
behaviour, which is why self-generated tests underperform provided ones. Three
constraints keep this an external yardstick: it is derived from the instruction
only (candidate code is never shown to the generator), generated once per task
and applied unchanged to every candidate, and generated at the BEST tier so the
bar is at least as strong as the model being judged. On top of that,
:func:`discriminates` discards any test the untouched baseline can pass.

A test that fails to compile or collect is inconclusive ("skip"), never a
verdict: our test being wrong must not condemn a correct candidate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rune.utils.logger import get_logger

log = get_logger(__name__)

_GEN_TIMEOUT_S = 60.0
_EVIDENCE_TAIL_CHARS = 600


@dataclass(frozen=True)
class Framework:
    """A project's test runner and where a generated test belongs."""

    name: str
    language: str
    test_path: str          # relative to the project root
    command: list[str]      # runs ONLY the generated test
    package: str = ""       # crate/module name the test imports


def detect_framework(cwd: str) -> Framework | None:
    """Identify the project's test framework from structured markers only.

    Returns None when no framework is recognized, so the caller falls back to
    the Evidence Gate rather than guessing.
    """
    try:
        entries = set(os.listdir(cwd))
    except OSError:
        return None

    if "Cargo.toml" in entries:
        package = "crate_under_test"
        try:
            with open(os.path.join(cwd, "Cargo.toml"), encoding="utf-8") as fh:
                in_pkg = False
                for line in fh:
                    s = line.strip()
                    if s.startswith("["):
                        in_pkg = s == "[package]"
                        continue
                    if in_pkg and s.startswith("name"):
                        _, _, raw = s.partition("=")
                        package = raw.strip().strip('"').replace("-", "_")
                        break
        except OSError:
            pass
        return Framework(
            name="cargo test", language="rust",
            test_path="tests/rune_verify.rs",
            command=["cargo", "test", "--test", "rune_verify"],
            package=package,
        )

    if {"pyproject.toml", "setup.py", "setup.cfg"} & entries or "tests" in entries:
        import sys
        return Framework(
            name="pytest", language="python",
            test_path="tests/test_rune_verify.py",
            command=[sys.executable, "-m", "pytest", "-q",
                     "tests/test_rune_verify.py"],
        )

    if "package.json" in entries:
        try:
            import json
            with open(os.path.join(cwd, "package.json"), encoding="utf-8") as fh:
                pkg = json.load(fh)
            dev = {**(pkg.get("devDependencies") or {}), **(pkg.get("dependencies") or {})}
            if "vitest" in dev:
                return Framework(
                    name="vitest", language="javascript",
                    test_path="rune_verify.test.js",
                    command=["npx", "vitest", "run", "rune_verify.test.js"],
                )
            if "jest" in dev:
                return Framework(
                    name="jest", language="javascript",
                    test_path="rune_verify.test.js",
                    command=["npx", "jest", "rune_verify.test.js"],
                )
        except (OSError, ValueError):
            pass
    return None


_MAX_API_LINES = 60


def extract_public_api(cwd: str, framework: Framework) -> str:
    """Public declarations (signatures only, no bodies) of the BASELINE tree.

    A contract alone often doesn't name enough API for a test to compile —
    measured: 4/5 generated tests failed to compile without this. Signatures are
    spec-level information, and they are read from the pre-edit baseline, never
    from a candidate, so no candidate's implementation choices leak into the bar
    it is judged by.
    """
    import re

    if framework.language == "rust":
        roots, exts = ["src"], (".rs",)
        pat = re.compile(r"^\s*pub\s+(fn|struct|enum|type|mod|use|const)\b.*")
    elif framework.language == "python":
        roots, exts = ["."], (".py",)
        pat = re.compile(r"^\s*(def|class)\s+[A-Za-z_].*")
    else:
        roots, exts = ["."], (".js", ".ts", ".mjs")
        pat = re.compile(r"^\s*export\s+(function|class|const)\b.*")

    lines: list[str] = []
    for root in roots:
        base = os.path.join(cwd, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                d for d in dirnames
                if d not in {"target", "node_modules", ".git", "__pycache__",
                             "tests", "test", ".venv"}
            ]
            for fn in sorted(filenames):
                if not fn.endswith(exts):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), cwd)
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                        found = [
                            m.group(0).strip().rstrip("{").strip()
                            for m in (pat.match(ln) for ln in fh)
                            if m
                        ]
                except OSError:
                    continue
                if found:
                    lines.append(f"// {rel}" if framework.language != "python"
                                 else f"# {rel}")
                    lines.extend(found)
                if len(lines) >= _MAX_API_LINES:
                    return "\n".join(lines[:_MAX_API_LINES])
    return "\n".join(lines)


_SYSTEM = (
    "You write ONE test file that checks whether an implementation satisfies a "
    "task's stated contract. You are given the task text and the project's "
    "EXISTING public declarations — never the implementation being judged. "
    "Write the test against the contract, not against any particular "
    "implementation you imagine.\n"
    "Rules:\n"
    "- Use the named test framework and that framework's idioms.\n"
    "- Test BEHAVIOUR stated in the contract: exact status codes, exact JSON "
    "shapes/values, ordering, error cases. One assertion per stated criterion.\n"
    "- Drive the code IN-PROCESS through the public API the contract names. "
    "NEVER start a server, bind a port, sleep, or make network calls — an "
    "in-process call is faster and cannot collide with other test runs.\n"
    "- Use ONLY the APIs named in the contract or listed in the declarations "
    "given to you. The test MUST compile against exactly those. If that is not "
    "enough to write a compiling test, output exactly NO_TEST.\n"
    "- The test must FAIL against the current code (the feature is not "
    "implemented yet) and pass only once the contract is satisfied. A test that "
    "passes either way is worthless.\n"
    "- No new dependencies beyond what the contract mentions.\n"
    "- The whole test must finish in well under a second.\n"
    "Output ONLY the test file body, or NO_TEST. No markdown fences, no prose."
)


async def generate_verification_test(
    instruction: str, framework: Framework, api_surface: str = ""
) -> str | None:
    """Write a contract-derived test, or None when the contract can't support one.

    The candidate's code is deliberately NOT an input; ``api_surface`` carries
    only baseline public declarations (see module docstring).
    """
    from rune.agent.evidence_gate import _strip_fences

    hint = (
        f"Test framework: {framework.name} ({framework.language}).\n"
        f"The test file will be saved as: {framework.test_path}\n"
    )
    if api_surface:
        hint += (
            "\nExisting public declarations in the project (signatures only — "
            "the bodies are NOT shown, and the code being judged is NOT shown):\n"
            f"{api_surface}\n"
        )
    if framework.package:
        hint += (
            f"The code under test is the crate `{framework.package}`; import it "
            f"as `{framework.package}::...` (this is an integration test in "
            f"tests/).\n"
        )
    try:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        client = get_llm_client()
        response = await client.completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{hint}\nTask:\n{instruction}\n\nTest file:"},
            ],
            tier=ModelTier.BEST,
            # A multi-endpoint contract needs a long test file; at 1200 it was
            # truncated mid-function and discarded as uncompilable (4/5 tasks).
            max_tokens=4000,
            timeout=_GEN_TIMEOUT_S,
        )
    except Exception as exc:  # pragma: no cover - network/SDK variance
        log.warning("generated_test_llm_failed", error=str(exc)[:120])
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

    body = _strip_fences(text)
    if not body or "NO_TEST" in body:
        log.info("generated_test_no_test")
        return None
    return body


# Output markers that mean "our test never ran", not "the artifact is wrong".
_INCONCLUSIVE_MARKERS = (
    "error[e",                 # rustc: type/name errors in OUR test
    "could not compile",
    "cannot find",
    "unresolved import",
    "modulenotfounderror",
    "importerror",
    "collection error",
    "cannot find module",
    "syntaxerror",
)


async def run_generated_test(
    test_body: str, framework: Framework, cwd: str
) -> tuple[str, str]:
    """Inject the test, run it, remove it. Returns ``(state, evidence)``.

    ``pass``  — the generated test ran and every assertion held.
    ``fail``  — it ran and an assertion failed; evidence carries the tail.
    ``skip``  — INCONCLUSIVE: the test could not compile/collect/run. Our own
                test being unusable must never condemn a candidate.
    """
    import asyncio

    path = os.path.join(cwd, framework.test_path)
    created_dir = None
    parent = os.path.dirname(path)
    try:
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
            created_dir = parent
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(test_body if test_body.endswith("\n") else test_body + "\n")
    except OSError as exc:
        log.warning("generated_test_write_failed", error=str(exc)[:120])
        return "skip", ""

    try:
        proc = await asyncio.create_subprocess_exec(
            *framework.command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_GEN_TIMEOUT_S
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            log.warning("generated_test_timeout", timeout_s=_GEN_TIMEOUT_S)
            return "skip", ""
        out = (stdout or b"").decode("utf-8", errors="replace")
        rc = proc.returncode
    except Exception as exc:  # pragma: no cover - spawn variance
        log.warning("generated_test_spawn_error", error=str(exc)[:120])
        return "skip", ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        if created_dir:
            try:
                os.rmdir(created_dir)  # only succeeds if we left it empty
            except OSError:
                pass

    if rc == 0:
        from rune.agent.auto_verify import assertions_ran

        # A generated test that asserted nothing is the very hole this module
        # was built to close — don't let it rubber-stamp a candidate.
        if assertions_ran(out) is False:
            log.info("generated_test_asserted_nothing")
            return "skip", ""
        return "pass", ""

    low = out.lower()
    if any(mark in low for mark in _INCONCLUSIVE_MARKERS):
        log.info("generated_test_inconclusive", rc=rc)
        return "skip", out[-_EVIDENCE_TAIL_CHARS:]
    return "fail", out[-_EVIDENCE_TAIL_CHARS:]


async def discriminates(
    test_body: str, framework: Framework, baseline_cwd: str
) -> bool:
    """Does this test distinguish "done" from "not done"?

    Run against the baseline — the tree before any candidate touched it, where
    the feature does not exist yet. The one disqualifying outcome is the
    baseline PASSING: measurement caught a generated test passing an
    implementation with the feature entirely absent, which would have reported
    "verified" on code that does nothing.

    A baseline that fails to compile still counts, and is the strongest signal:
    the test references API the task asks the candidate to create, so it cannot
    pass before the work is done. Demanding a clean ``fail`` instead rejected
    every API-adding task (3 of 5 measured).

    A merely broken test is not a risk — it fails to build against the candidate
    too, yielding ``skip``, which decides nothing.
    """
    state, _ = await run_generated_test(test_body, framework, baseline_cwd)
    if state == "pass":
        log.info("generated_test_not_discriminating", baseline_state=state)
        return False
    return True
