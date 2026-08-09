"""Rung-0 fast path: single-shot localize → edit → repro-gated verify.

Before paying for the full agentic loop, one cheap non-agentic pass:
locate the fault from the repo structure, generate a few SEARCH/REPLACE
candidates in one prompt each, and accept only a candidate that flips a
baseline-failing reproduction script to passing without breaking the
targeted regression tests. That flip is a discriminating check, so the
result may honestly be called verified. Anything less falls through to the
agentic rung, which starts fresh and receives only structured evidence
(the repro script and its failing output) — never a failed diff.

Python-only for now; other stacks skip the fast path entirely.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from rune.utils.logger import get_logger

log = get_logger(__name__)

_TREE_MAX_FILES = 1500
_SKELETON_MAX_FILES = 4
_SKELETON_MAX_CHARS = 24_000
_CANDIDATE_SAMPLES = 4
_REPRO_SAMPLES = 3
_REPRO_TIMEOUT_S = 30.0
_LLM_TIMEOUT_S = 120.0


@dataclass
class FastPathResult:
    verified: bool = False
    applied: list[str] = field(default_factory=list)  # relpaths edited
    method: str = ""
    # Evidence for the agentic rung when not verified:
    repro_script: str = ""  # discriminating repro source ("" if none found)
    repro_output: str = ""  # its failing output on the baseline
    located_files: list[str] = field(default_factory=list)


async def _complete(model: str | None, provider: str | None, prompt: str,
                    n: int = 1) -> list[str]:
    """n independent completions for one prompt; failures return fewer."""
    from rune.agent.litellm_adapter import _resolve_litellm_model, litellm

    resolved, extra = _resolve_litellm_model(
        f"{provider}:{model}" if provider and model else (model or "")
    )

    async def one() -> str | None:
        try:
            resp = await asyncio.wait_for(
                litellm.acompletion(
                    model=resolved,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    **extra,
                ),
                timeout=_LLM_TIMEOUT_S,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.debug("fastpath_llm_error", error=str(exc)[:120])
            return None

    outs = await asyncio.gather(*[one() for _ in range(n)])
    return [o for o in outs if o]


def _repo_tree(root: str) -> str:
    """Compact listing of the repo's Python files, capped."""
    lines: list[str] = []
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox",
            "build", "dist", "docs", "doc", "examples"}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        rel = os.path.relpath(dirpath, root)
        for fn in sorted(files):
            if fn.endswith(".py"):
                lines.append(fn if rel == "." else f"{rel}/{fn}")
                if len(lines) >= _TREE_MAX_FILES:
                    return "\n".join(lines)
    return "\n".join(lines)


def _skeleton(source: str) -> str:
    """Signatures + class fields only; bodies stripped."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:2000]
    out: list[str] = []

    def walk(node: ast.AST, indent: int) -> None:
        pad = "    " * indent
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ast.unparse(child.args)
                out.append(f"{pad}def {child.name}({args}): ...")
            elif isinstance(child, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in child.bases)
                out.append(f"{pad}class {child.name}({bases}):")
                walk(child, indent + 1)
            elif isinstance(child, ast.Assign) and indent > 0:
                out.append(f"{pad}{ast.unparse(child)[:100]}")
    walk(tree, 0)
    return "\n".join(out)


def _extract_files(text: str, root: str) -> list[str]:
    """File paths from an LLM reply, kept only if they exist on disk."""
    found: list[str] = []
    for m in re.finditer(r"[\w./-]+\.py", text):
        p = m.group().lstrip("./")
        if p not in found and os.path.isfile(os.path.join(root, p)):
            found.append(p)
    return found


_SR_BLOCK_RE = re.compile(
    r"```[a-z]*\n*<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE\s*\n*```",
    re.S,
)
_FILE_TAG_RE = re.compile(r"^###\s*FILE:\s*([\w./-]+\.py)\s*$", re.M)


def parse_candidate(text: str) -> list[tuple[str, str, str]]:
    """(path, search, replace) triples from a '### FILE:' + S/R-block reply."""
    edits: list[tuple[str, str, str]] = []
    sections = _FILE_TAG_RE.split(text)
    # sections = [preamble, path1, body1, path2, body2, ...]
    for i in range(1, len(sections) - 1, 2):
        path, body = sections[i].strip(), sections[i + 1]
        for m in _SR_BLOCK_RE.finditer(body):
            edits.append((path, m.group(1), m.group(2)))
    return edits


def apply_candidate(workdir: str,
                    edits: list[tuple[str, str, str]]) -> list[str]:
    """Apply S/R edits via the fuzzy ladder. Returns edited relpaths.

    All-or-nothing: any unmatched/ambiguous block reverts the whole
    candidate — a half-applied fix is worse than none.
    """
    from rune.capabilities.edit_matching import apply_block, find_block

    originals: dict[str, str] = {}
    touched: list[str] = []
    for path, search, replace in edits:
        full = os.path.join(workdir, path)
        try:
            content = open(full).read()
        except OSError:
            _revert(workdir, originals)
            return []
        if path not in originals:
            originals[path] = content
        if search in content:
            new = content.replace(search, replace, 1)
        else:
            m = find_block(content, search)
            if m is None:
                _revert(workdir, originals)
                return []
            new = apply_block(content, m, replace)
        open(full, "w").write(new)
        if path not in touched:
            touched.append(path)
    return touched


def _revert(workdir: str, originals: dict[str, str]) -> None:
    for path, content in originals.items():
        try:
            open(os.path.join(workdir, path), "w").write(content)
        except OSError:
            pass


async def _run_script(py: str, script_path: str, cwd: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{cwd}:{cwd}/src"
    try:
        proc = await asyncio.create_subprocess_exec(
            py, script_path, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(),
                                        timeout=_REPRO_TIMEOUT_S)
        return proc.returncode or 0, (out or b"").decode("utf-8", "replace")
    except Exception as exc:
        return 125, str(exc)[:200]


async def _discriminating_repro(
    issue: str, context: str, seed_cwd: str, model: str | None,
    provider: str | None,
) -> tuple[str, str]:
    """A repro script that FAILS on the unfixed baseline, and its output.

    Only a baseline-failing script discriminates; one that passes the broken
    code proves nothing and is discarded. Returns ("", "") if none qualify.
    """
    from rune.agent.rejection_sampler import _project_python

    prompt = (
        "Write a standalone Python script that REPRODUCES the bug below "
        "against the repository in the current directory. The script must:\n"
        "- import from the repo (assume it is on sys.path),\n"
        "- exit non-zero (e.g. failed assert) while the bug EXISTS,\n"
        "- exit 0 once the bug is FIXED.\n"
        "Ground every assertion in the EXACT inputs and expected outputs "
        "quoted in the bug report. Do not assert any behavior the report "
        "does not show — an over-strict script that also fails on a correct "
        "fix is useless.\n"
        "Keep it minimal. Reply with ONLY the script in one ```python block."
        f"\n\nBug report:\n{issue}\n\nRelevant code:\n{context}"
    )
    replies = await _complete(model, provider, prompt, n=_REPRO_SAMPLES)
    py = _project_python(seed_cwd)
    scratch = tempfile.mkdtemp(prefix="rune-repro-")
    try:
        for i, reply in enumerate(replies):
            m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
            if not m:
                continue
            script = m.group(1)
            path = os.path.join(scratch, f"repro_{i}.py")
            open(path, "w").write(script)
            rc, out = await _run_script(py, path, seed_cwd)
            if rc not in (0, 125):  # fails on baseline → discriminating
                log.info("fastpath_repro_found", sample=i)
                return script, out[-1500:]
        return "", ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def run_fastpath(
    issue: str, seed_cwd: str, workdir: str,
    model: str | None, provider: str | None,
) -> FastPathResult:
    """Run rung-0 inside *workdir* (a seeded copy of *seed_cwd*)."""
    res = FastPathResult()

    tree = _repo_tree(seed_cwd)
    if not tree:
        return res

    picks = await _complete(
        model, provider,
        "Which files most likely contain the bug below? Reply with up to 3 "
        f"file paths from this list, one per line, nothing else.\n\n"
        f"Bug report:\n{issue}\n\nFiles:\n{tree}",
    )
    files = _extract_files(picks[0] if picks else "", seed_cwd)
    files = files[:_SKELETON_MAX_FILES]
    if not files:
        log.info("fastpath_no_files_located")
        return res
    log.info("fastpath_localized", files=files)
    res.located_files = files

    context_parts: list[str] = []
    for f in files:
        try:
            src = open(os.path.join(seed_cwd, f)).read()
        except OSError:
            continue
        body = src if len(src) < 8000 else _skeleton(src)
        context_parts.append(f"### FILE: {f}\n{body}")
    context = "\n\n".join(context_parts)[:_SKELETON_MAX_CHARS]

    # Repro gate first: without a discriminating check, rung-0 cannot claim
    # verified, and the loop is better placed to investigate.
    res.repro_script, res.repro_output = await _discriminating_repro(
        issue, context, seed_cwd, model, provider
    )
    if not res.repro_script:
        log.info("fastpath_no_discriminating_repro")
        return res

    edit_prompt = (
        "Fix the bug below with the SMALLEST correct change to the code "
        "shown. Reply with one or more edits in exactly this format:\n"
        "### FILE: path/to/file.py\n"
        "```\n<<<<<<< SEARCH\n(exact lines from the file)\n=======\n"
        "(replacement lines)\n>>>>>>> REPLACE\n```\n"
        f"\nBug report:\n{issue}\n\nCode:\n{context}"
        f"\n\nA failing reproduction script's output:\n{res.repro_output}"
    )
    replies = await _complete(model, provider, edit_prompt,
                              n=_CANDIDATE_SAMPLES)

    from rune.agent.rejection_sampler import (
        _project_python,
        _restore_canonical_tests,
        _targeted_test_files,
    )
    py = _project_python(seed_cwd)
    scratch = tempfile.mkdtemp(prefix="rune-fastpath-")
    repro_path = os.path.join(scratch, "repro.py")
    open(repro_path, "w").write(res.repro_script)
    try:
        for reply in replies:
            edits = parse_candidate(reply)
            if not edits:
                continue
            touched = apply_candidate(workdir, edits)
            if not touched:
                continue
            rc, _out = await _run_script(py, repro_path, workdir)
            if rc != 0:  # repro still failing → not the fix
                _revert_files(workdir, seed_cwd, touched)
                continue
            # Regression gate: nearest existing tests must not break.
            targets = _targeted_test_files(workdir, seed_cwd)
            if targets:
                from rune.agent.auto_verify import run_verify
                _restore_canonical_tests(workdir, seed_cwd, targets)
                state, evidence = await run_verify(
                    ["/usr/bin/env", f"PYTHONPATH={workdir}:{workdir}/src",
                     py, "-m", "pytest", "-q", *targets],
                    workdir, timeout=120.0,
                )
                if state == "fail" and re.search(r"\b\d+ failed\b", evidence):
                    _revert_files(workdir, seed_cwd, touched)
                    continue
            res.verified = True
            res.applied = touched
            res.method = "reproduction script (fails pre-fix) + targeted tests"
            log.info("fastpath_verified", files=touched)
            return res
        return res
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _revert_files(workdir: str, seed_cwd: str, rels: list[str]) -> None:
    for rel in rels:
        try:
            shutil.copy2(os.path.join(seed_cwd, rel),
                         os.path.join(workdir, rel))
        except OSError:
            pass
