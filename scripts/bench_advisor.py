#!/usr/bin/env python3
"""Micro-benchmark for the advisor layer.

Runs the SAME short coding task through ``NativeAgentLoop`` twice:

    Run A: Haiku executor, no advisor (baseline)
    Run B: Haiku executor + Opus advisor

Collects per-run metrics (steps, tokens, duration, success, advisor
call count) and prints a side-by-side comparison. Uses an isolated
temp HOME so the real RUNE state is never touched, and copies the
real ``~/.rune/.env`` + ``config.yaml`` so API keys flow through.

This is a demonstration, not a rigorous benchmark — a single sample
per run is inherently noisy. The goal is to prove the plumbing works
end-to-end and surface any integration regressions.

Usage:
    uv run python scripts/bench_advisor.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# isolation (must happen before any rune import)

# Variant is set via RUNE_BENCH_VARIANT env: "easy" (5 tests, default budget)
# or "big" (10 tests, 2x token budget). Default preserves original behavior.
_VARIANT = os.environ.get("RUNE_BENCH_VARIANT", "").strip() or "default"
_WS_SUFFIX = f"-{_VARIANT}" if _VARIANT != "default" else ""

TMPDIR = tempfile.mkdtemp(prefix=f"rune-advisor-bench-{_VARIANT}-")
REAL_HOME = Path.home()
_WORKSPACE = REAL_HOME / f".rune-advisor-bench-workspace{_WS_SUFFIX}"
# Fresh workspace to avoid cross-variant contamination
if _WORKSPACE.exists():
    shutil.rmtree(_WORKSPACE)
_WORKSPACE.mkdir(parents=True, exist_ok=True)

# Copy config + .env + google-credentials so LiteLLM picks up the keys.
# Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS to point at a
# reachable service-account JSON — RUNE's loader auto-discovers
# rune_home()/google-credentials.json, which here is TMPDIR/.rune/.
_cfg_dst = Path(TMPDIR) / ".rune"
_cfg_dst.mkdir(parents=True, exist_ok=True)
for fname in ("config.yaml", ".env", "google-credentials.json"):
    src = REAL_HOME / ".rune" / fname
    if src.exists():
        shutil.copy2(src, _cfg_dst / fname)
# RUNE also auto-discovers cwd/.rune/google-credentials.json (project
# local). Check that path too as a fallback.
_project_creds = Path("/Users/gmldns46/workspace/rune/.rune/google-credentials.json")
if _project_creds.exists() and not (_cfg_dst / "google-credentials.json").exists():
    shutil.copy2(_project_creds, _cfg_dst / "google-credentials.json")

# Symlink embedding models to avoid re-download
_real_models = REAL_HOME / ".rune" / "data" / "models"
if _real_models.exists():
    _dst = Path(TMPDIR) / ".rune" / "data" / "models"
    _dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(_real_models, _dst)
    except FileExistsError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rune.utils.paths as _paths  # noqa: E402

_paths._HOME = Path(TMPDIR)

# imports after isolation

import asyncio  # noqa: E402
import contextlib  # noqa: E402

from rune.agent.advisor import policy as _policy_mod  # noqa: E402
from rune.agent.advisor import service as _advisor_svc_mod  # noqa: E402
from rune.agent.advisor.service import AdvisorService  # noqa: E402
from rune.agent.loop import NativeAgentLoop  # noqa: E402
from rune.types import AgentConfig  # noqa: E402

# Track advisor services created during a run so we can read their
# budget after the loop finishes.
_captured_services: list[AdvisorService] = []
_original_for_episode = AdvisorService.for_episode


def _tracking_for_episode(executor_model: str) -> AdvisorService:
    svc = _original_for_episode(executor_model)
    _captured_services.append(svc)
    return svc


_advisor_svc_mod.AdvisorService.for_episode = staticmethod(_tracking_for_episode)
AdvisorService.for_episode = staticmethod(_tracking_for_episode)  # re-export

# Instrument should_call so we can count policy evaluations even when
# no trigger fires. This gives us visibility into "the advisor layer
# was considered but the task was easy" vs "it was never reached".
_policy_checks: list[tuple[str, str]] = []
_original_should_call = _policy_mod.should_call


def _instrumented_should_call(state, inp):
    call, trigger = _original_should_call(state, inp)
    _policy_checks.append(
        (
            "fire" if call else "skip",
            str(trigger or "-"),
        )
    )
    return call, trigger


_policy_mod.should_call = _instrumented_should_call
# loop_integration imports `should_call` by name, so also patch there
from rune.agent.advisor import loop_integration as _li_mod  # noqa: E402

_li_mod.should_call = _instrumented_should_call

# task definition

# Executor + advisor are overridable via env to allow running the same
# bench across provider pairs without editing the script.
EXECUTOR_PROVIDER = os.environ.get("RUNE_BENCH_EXECUTOR_PROVIDER", "ollama")
EXECUTOR_MODEL = os.environ.get("RUNE_BENCH_EXECUTOR_MODEL", "gemma4:e2b")
ADVISOR_MODEL = os.environ.get(
    "RUNE_BENCH_ADVISOR_MODEL", "ollama/qwen3-coder:480b-cloud",
)

# Task pool — one per variant
_TASK_BIG = (
    "Create {workspace}/timeparse.py with a function `parse_time(s: str) -> int` "
    "that parses duration strings into total seconds. Rules (all MUST be enforced):\n"
    "  - Valid tokens: '<n>h', '<n>m', '<n>s' where <n> is a positive integer.\n"
    "  - If multiple tokens are present, the order MUST be h → m → s. "
    "Out-of-order MUST raise ValueError('invalid order').\n"
    "  - At least one token is required. Empty string raises ValueError('empty').\n"
    "  - Leading/trailing whitespace is allowed and trimmed.\n"
    "  - Any unknown characters / malformed input raises ValueError('malformed').\n"
    "  - Negative numbers raise ValueError('negative not allowed').\n"
    "\n"
    "Then add a main block that runs EXACTLY these 10 checks and prints "
    "'PASS: <id>' or 'FAIL: <id> msg=<msg>' for each. At the end print "
    "'RESULT: <passed>/10'.\n"
    "  1. parse_time('1h30m') == 5400\n"
    "  2. parse_time('  45m  ') == 2700  # trimmed\n"
    "  3. parse_time('2h') == 7200\n"
    "  4. parse_time('90s') == 90\n"
    "  5. parse_time('1h2m3s') == 3723\n"
    "  6. parse_time('') raises ValueError('empty')\n"
    "  7. parse_time('abc') raises ValueError('malformed')\n"
    "  8. parse_time('30m1h') raises ValueError('invalid order')\n"
    "  9. parse_time('-5m') raises ValueError('negative not allowed')\n"
    " 10. parse_time('1s2m') raises ValueError('invalid order')\n"
    "\n"
    "Run `python {workspace}/timeparse.py` and verify the output shows "
    "'RESULT: 10/10'. If fewer pass, read the FAIL lines, fix timeparse.py, "
    "and re-run until all 10 pass.\n"
)

_TASK_EASY = (
    "Create {workspace}/timeparse.py with a function `parse_time(s: str) -> int` "
    "that parses duration strings into total seconds.\n"
    "  - Valid tokens: '<n>h', '<n>m', '<n>s' where <n> is a non-negative integer.\n"
    "  - Multiple tokens concatenated (e.g. '1h30m') sum up.\n"
    "  - Empty string raises ValueError('empty').\n"
    "  - Unknown characters raise ValueError('malformed').\n"
    "\n"
    "Then add a main block that runs EXACTLY these 5 checks and prints "
    "'PASS: <id>' or 'FAIL: <id> msg=<msg>' for each. At the end print "
    "'RESULT: <passed>/5'.\n"
    "  1. parse_time('1h30m') == 5400\n"
    "  2. parse_time('2h') == 7200\n"
    "  3. parse_time('90s') == 90\n"
    "  4. parse_time('') raises ValueError('empty')\n"
    "  5. parse_time('abc') raises ValueError('malformed')\n"
    "\n"
    "Run `python {workspace}/timeparse.py` and verify the output shows "
    "'RESULT: 5/5'. If fewer pass, read the FAIL lines, fix timeparse.py, "
    "and re-run until all 5 pass.\n"
)

_TASK_MEDIUM = (
    "Create {workspace}/urlparse_lite.py with a function "
    "`parse_url(url: str) -> dict` that parses HTTP(S) URLs into a dict with "
    "keys: scheme, host, port, path, query, fragment.\n\n"
    "CRITICAL CONSTRAINT: Do NOT import the `urllib` package or any of its "
    "submodules (urllib.parse, urllib.request, etc.). Do NOT use `requests`. "
    "You MAY import `re`. Implement the parser manually.\n\n"
    "Rules (all MUST be enforced):\n"
    "  - scheme: 'http' or 'https' only. ValueError('unsupported scheme: <s>') otherwise. "
    "Stored lowercase even if input is 'HTTPS://...'.\n"
    "  - port: int. Default 80 for http, 443 for https when omitted. "
    "If explicit, must be 1..65535 else ValueError('invalid port').\n"
    "  - path: starts with '/'. Default '/' if empty.\n"
    "  - query: dict[str, str]. Empty dict {{}} if no query string. Keys and "
    "values MUST be percent-decoded ('%20' -> ' ', '%2B' -> '+').\n"
    "  - fragment: the part after '#', or None if absent.\n"
    "  - userinfo: 'user:pass@host' is allowed but MUST be stripped from "
    "host. host stores just the hostname.\n"
    "  - An empty or malformed URL raises ValueError.\n\n"
    "Then add a main block that runs EXACTLY these 8 checks. For each check "
    "print 'PASS: <id>' or 'FAIL: <id> got=<got> expected=<exp>'. At the end "
    "print 'RESULT: <passed>/8'.\n"
    "  1. parse_url('http://example.com')['host'] == 'example.com'\n"
    "  2. parse_url('http://example.com')['port'] == 80               # default http port as int\n"
    "  3. parse_url('https://example.com:8080/api')['port'] == 8080\n"
    "  4. parse_url('HTTPS://Example.com/')['scheme'] == 'https'      # normalized lowercase\n"
    "  5. parse_url('http://x.com/p?a=1&b=2')['query'] == {{'a':'1','b':'2'}}\n"
    "  6. parse_url('http://x.com/p?msg=hello%20world')['query']['msg'] == 'hello world'  # percent decode\n"
    "  7. parse_url('http://user:pass@x.com/path')['host'] == 'x.com' # userinfo stripped\n"
    "  8. parse_url('ftp://x.com') raises ValueError                   # unsupported scheme\n\n"
    "Run `python {workspace}/urlparse_lite.py` and verify 'RESULT: 8/8'. "
    "If any FAIL, read the error, fix the file, re-run until all 8 pass.\n"
)

# Goldilocks variant: a parser that has SPECIFIC traps weak models trip on
# but can recover from with targeted advisor guidance.
_TASK_COLOR = (
    "Create {workspace}/colorparse.py with a function "
    "`parse_color(s: str) -> tuple[int, int, int]` that parses CSS-like color "
    "strings into (r, g, b) integer tuples (each 0..255).\n\n"
    "Supported formats:\n"
    "  - Long hex: '#RRGGBB' (e.g. '#FF8000' -> (255, 128, 0)). Case-insensitive.\n"
    "  - SHORT hex: '#RGB' where each digit is duplicated. "
    "EXAMPLE: '#F0A' MUST become (0xFF, 0x00, 0xAA) = (255, 0, 170). "
    "DO NOT just pad with zeros — duplicate the digit.\n"
    "  - rgb(): 'rgb(R, G, B)' where each is decimal 0..255 (whitespace allowed).\n"
    "  - Named: 'red'=(255,0,0), 'green'=(0,128,0), 'blue'=(0,0,255), "
    "'white'=(255,255,255), 'black'=(0,0,0). Case-insensitive.\n"
    "  - Anything else raises ValueError('invalid color: <s>').\n\n"
    "Then add a main block that runs EXACTLY these 6 checks and prints "
    "'PASS: <id>' or 'FAIL: <id> got=<got> expected=<exp>' for each. At the end "
    "print 'RESULT: <passed>/6'.\n"
    "  1. parse_color('#FF8000') == (255, 128, 0)\n"
    "  2. parse_color('#F0A') == (255, 0, 170)        # short hex doubling\n"
    "  3. parse_color('#fa0') == (255, 170, 0)        # short hex case-insensitive\n"
    "  4. parse_color('rgb(100, 200, 50)') == (100, 200, 50)\n"
    "  5. parse_color('RED') == (255, 0, 0)            # named, case-insensitive\n"
    "  6. parse_color('not_a_color') raises ValueError\n\n"
    "Run `python {workspace}/colorparse.py` and verify the output shows "
    "'RESULT: 6/6'. If fewer pass, read the FAIL lines (especially the actual "
    "vs expected values), fix colorparse.py, and re-run until all 6 pass.\n"
)

if _VARIANT == "easy":
    TASK = _TASK_EASY
elif _VARIANT == "color":
    TASK = _TASK_COLOR
elif _VARIANT == "medium":
    TASK = _TASK_MEDIUM
else:
    TASK = _TASK_BIG

# "big" variant doubles the token budget so the executor has a real chance to
# finish. Applied by monkey-patching the intent→budget map before loop imports
# pick it up via `from rune.agent.loop import ...`.
if _VARIANT == "big":
    try:
        import rune.agent.loop as _loop_mod  # noqa: E402
        _orig_budget = getattr(_loop_mod, "_BUDGET_BY_INTENT", None)
        if isinstance(_orig_budget, dict):
            for k in list(_orig_budget.keys()):
                _orig_budget[k] = _orig_budget[k] * 2
    except Exception:
        pass


# metrics

@dataclass
class RunMetrics:
    name: str
    trace_reason: str = ""
    steps: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    success: bool = False
    advisor_enabled: bool = False
    advisor_calls: int = 0
    advisor_tokens_in: int = 0
    advisor_tokens_out: int = 0
    advisor_disabled_reason: str | None = None
    answer_preview: str = ""
    error: str | None = None
    gate_blocked_count: int = 0
    files_created: list[str] = field(default_factory=list)
    policy_checks: int = 0
    policy_fires: int = 0
    policy_trigger_summary: dict[str, int] = field(default_factory=dict)


async def _run_once(name: str, advisor_env: str | None) -> RunMetrics:
    """Execute a single turn. ``advisor_env`` is the value of the
    ``RUNE_ADVISOR_MODEL`` env var for this run (``None`` = disable)."""
    global _captured_services, _policy_checks
    _captured_services = []
    _policy_checks = []

    # Env: scope advisor config to this run.
    prev_env = os.environ.get("RUNE_ADVISOR_MODEL")
    if advisor_env:
        os.environ["RUNE_ADVISOR_MODEL"] = advisor_env
    else:
        os.environ.pop("RUNE_ADVISOR_MODEL", None)

    metrics = RunMetrics(name=name, advisor_enabled=bool(advisor_env))

    # Clean workspace for each run so they start identical.
    for path in _WORKSPACE.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception:
            pass

    # Reload config so active provider/model takes effect
    from rune.config import get_config
    from rune.config.loader import reset_config
    reset_config()
    cfg = get_config()
    cfg.llm.active_provider = EXECUTOR_PROVIDER
    cfg.llm.active_model = EXECUTOR_MODEL
    cfg.llm.default_provider = EXECUTOR_PROVIDER

    config = AgentConfig()
    config.model = EXECUTOR_MODEL
    config.provider = EXECUTOR_PROVIDER
    loop = NativeAgentLoop(config=config)

    collected: list[str] = []

    async def _on_text(delta: str) -> None:
        collected.append(delta)

    loop.on("text_delta", _on_text)

    async def _auto_approve(_cap: str, _reason: str) -> bool:
        return True

    loop.set_approval_callback(_auto_approve)

    goal = TASK.replace("{workspace}", str(_WORKSPACE))
    t0 = time.monotonic()
    try:
        trace = await loop.run(
            goal,
            context={"workspace_root": str(_WORKSPACE)},
        )
        metrics.trace_reason = trace.reason or "unknown"
        metrics.steps = trace.final_step or 0
        metrics.tokens = getattr(trace, "total_tokens_used", 0)
        metrics.success = metrics.trace_reason == "completed"
        metrics.gate_blocked_count = getattr(loop, "_gate_blocked_count", 0)
    except Exception as exc:
        metrics.error = f"{type(exc).__name__}: {exc}"
    finally:
        metrics.duration_s = time.monotonic() - t0
        metrics.answer_preview = ("".join(collected))[:200].replace("\n", " ")
        # Restore env
        if prev_env is not None:
            os.environ["RUNE_ADVISOR_MODEL"] = prev_env
        else:
            os.environ.pop("RUNE_ADVISOR_MODEL", None)

    # Read advisor service state if any were captured
    if _captured_services:
        svc = _captured_services[-1]
        metrics.advisor_calls = svc.budget.calls_used
        metrics.advisor_tokens_out = svc.budget.tokens_used
        metrics.advisor_disabled_reason = svc.budget.disabled_reason

    # Policy evaluation summary (how often should_call was consulted)
    metrics.policy_checks = len(_policy_checks)
    metrics.policy_fires = sum(1 for r, _ in _policy_checks if r == "fire")
    trig_summary: dict[str, int] = {}
    for r, t in _policy_checks:
        if r == "fire":
            trig_summary[t] = trig_summary.get(t, 0) + 1
    metrics.policy_trigger_summary = trig_summary

    # Gather created files
    try:
        metrics.files_created = sorted(
            str(p.relative_to(_WORKSPACE)) for p in _WORKSPACE.rglob("*")
            if p.is_file()
        )[:10]
    except Exception:
        pass

    return metrics


def _fmt(v: float, unit: str = "") -> str:
    if isinstance(v, float):
        return f"{v:.1f}{unit}"
    return f"{v}{unit}"


def _print_comparison(baseline: RunMetrics, advised: RunMetrics) -> None:
    rows = [
        ("trace_reason",   baseline.trace_reason, advised.trace_reason),
        ("success",        str(baseline.success), str(advised.success)),
        ("steps",          str(baseline.steps), str(advised.steps)),
        ("tokens (exec)",  str(baseline.tokens), str(advised.tokens)),
        ("duration",       f"{baseline.duration_s:.1f}s", f"{advised.duration_s:.1f}s"),
        ("gate_blocked",   str(baseline.gate_blocked_count), str(advised.gate_blocked_count)),
        ("policy_checks",  str(baseline.policy_checks), str(advised.policy_checks)),
        ("policy_fires",   str(baseline.policy_fires), str(advised.policy_fires)),
        ("triggers",       str(baseline.policy_trigger_summary) or "-",
                           str(advised.policy_trigger_summary) or "-"),
        ("advisor_calls",  str(baseline.advisor_calls), str(advised.advisor_calls)),
        ("advisor_tokens", str(baseline.advisor_tokens_out), str(advised.advisor_tokens_out)),
        ("advisor_off?",   baseline.advisor_disabled_reason or "-",
                           advised.advisor_disabled_reason or "-"),
        ("files_created",  str(len(baseline.files_created)), str(len(advised.files_created))),
    ]

    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows) + 2
    col3 = max(len(r[2]) for r in rows) + 2
    print()
    print("=" * (col1 + col2 + col3 + 6))
    print(f"  {'metric':<{col1}}  {'baseline (no adv)':<{col2}}  {'+ advisor (opus)':<{col3}}")
    print("-" * (col1 + col2 + col3 + 6))
    for k, a, b in rows:
        print(f"  {k:<{col1}}  {a:<{col2}}  {b:<{col3}}")
    print("=" * (col1 + col2 + col3 + 6))

    if baseline.error:
        print(f"\nbaseline error: {baseline.error}")
    if advised.error:
        print(f"\nadvised  error: {advised.error}")

    print(f"\nbaseline answer preview: {baseline.answer_preview}")
    print(f"advised  answer preview: {advised.answer_preview}")
    print(f"\nbaseline files: {baseline.files_created}")
    print(f"advised  files: {advised.files_created}")


async def main() -> None:
    print(f"[bench] tmpdir:    {TMPDIR}", flush=True)
    print(f"[bench] workspace: {_WORKSPACE}", flush=True)
    print(f"[bench] executor:  {EXECUTOR_PROVIDER}/{EXECUTOR_MODEL}", flush=True)
    print(f"[bench] advisor:   {ADVISOR_MODEL}", flush=True)
    print(flush=True)

    print("[bench] Run A: baseline (no advisor)...", flush=True)
    baseline = await _run_once("baseline", advisor_env=None)
    print(
        f"  → reason={baseline.trace_reason} steps={baseline.steps} "
        f"duration={baseline.duration_s:.1f}s",
        flush=True,
    )

    print(f"[bench] Run B: with advisor ({ADVISOR_MODEL})...", flush=True)
    advised = await _run_once("advised", advisor_env=ADVISOR_MODEL)
    print(
        f"  → reason={advised.trace_reason} steps={advised.steps} "
        f"duration={advised.duration_s:.1f}s advisor_calls={advised.advisor_calls}",
        flush=True,
    )

    _print_comparison(baseline, advised)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
