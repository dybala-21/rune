#!/usr/bin/env python3
"""Run arithmetic, web, CSV and code checks with Grok, Gemini, then Claude.

Each run uses a temporary workspace and the configured credentials.
Usage: .venv/bin/python scripts/e2e_core.py --output /tmp/rune-core-results.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODELS = (("xai", "grok-4.6"), ("gemini", "gemini-2.5-flash"), ("anthropic", "claude-opus-5"))
SCENARIOS = ("arithmetic", "decimal", "web", "csv", "code")
CSV = "department,amount\nsales,1200\nengineering,850\nsales,-200\nengineering,150\nsupport,450\n"
CODE = "def average(values):\n    return sum(values) / (len(values) + 1)\n"
CHECKS = '''import unittest
from stats import average

class AverageTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(average([2, 4, 6]), 4)
    def test_single(self):
        self.assertEqual(average([7]), 7)
    def test_negative(self):
        self.assertEqual(average([-2, 2]), 0)
    def test_empty(self):
        with self.assertRaises(ValueError):
            average([])
'''


def prepare(case: str, workspace: Path) -> str:
    if case == "web":
        return ("https://docs.x.ai/developers/model-capabilities/text/reasoning 를 읽고 "
                "Grok 4.6의 기본 추론 수준과 low의 용도를 한국어 두 문장으로 설명해. 근거 링크도 붙여줘.")
    if case == "arithmetic":
        return "173 × 29 − 417의 값을 숫자만으로 답해줘."
    if case == "decimal":
        return "Evaluate 19.95 × 7 - 12.4. Reply with just the numeric result."
    if case == "csv":
        (workspace / "expenses.csv").write_text(CSV)
        return ("expenses.csv를 읽고 department별 amount 합계를 department 오름차순으로 "
                "summary.csv에 저장해. 헤더는 department,total로 하고 음수도 합산해. "
                "저장한 값을 확인한 뒤 결과를 짧게 알려줘.")
    (workspace / "stats.py").write_text(CODE)
    (workspace / "test_stats.py").write_text(CHECKS)
    return ("stats.py의 average 버그를 수정해. 빈 입력은 ValueError로 처리해야 해. "
            "test_stats.py는 수정하지 말고 python3 -m unittest -v로 수정 전후를 검증한 뒤 "
            "실패했던 테스트와 수정 결과를 정확하게 요약해.")


def verify(case: str, workspace: Path, answer: str) -> dict:
    if case == "web":
        return {"answer": "high" in answer.lower() and "low" in answer.lower()
                and "https://docs.x.ai/developers/model-capabilities/text/reasoning" in answer,
                "no_files": not list(workspace.glob("*"))}
    if case == "arithmetic":
        return {"answer": answer.strip().replace(",", "") == "4600"}
    if case == "decimal":
        return {"answer": answer.strip() == "127.25"}
    if case == "csv":
        path = workspace / "summary.csv"
        if not path.is_file():
            return {"artifact": False}
        try:
            with path.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                header = reader.fieldnames
            values = [(r["department"], float(r["total"])) for r in rows]
            return {"artifact": header == ["department", "total"] and values == [
                ("engineering", 1000), ("sales", 1000), ("support", 450)],
                "source_unchanged": (workspace / "expenses.csv").read_text() == CSV}
        except (KeyError, ValueError, OSError):
            return {"artifact": False}
    result = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=workspace,
                            capture_output=True, text=True, timeout=15)
    return {"tests": result.returncode == 0,
            "tests_unchanged": (workspace / "test_stats.py").read_text() == CHECKS}


def verified_code_comparison(state) -> bool:
    from rune.agent.test_claims import comparison_evidence

    before = {f"test_stats.AverageTests.test_{name}": "pass" if name == "negative" else "fail"
              for name in ("empty", "negative", "positive", "single")}
    after = dict.fromkeys(before, "pass")
    # Added regression tests may increase the total; the original cases must remain.
    return state.passed and any(
        all(pair[phase]["complete"] and expected.items() <= {
            case["identity"]: case["status"] for case in pair[phase]["cases"]
        }.items() for phase, expected in (("before", before), ("after", after)))
        and pair["after"]["failed_tests"] == 0
        for pair in comparison_evidence(state))


async def worker(case: str, provider: str, model: str, result_path: Path, routing: str = "connected") -> None:
    sys.path.insert(0, str(ROOT))
    from rune.config import get_config
    from rune.utils.logger import configure_logging

    configure_logging(level="WARNING")
    cfg = get_config()
    cfg.llm.active_provider = cfg.llm.default_provider = provider
    cfg.llm.active_model = cfg.llm.default_model = model
    cfg.llm.route_simple_queries = False
    cfg.llm.decision_routing.backend = routing
    # Use the same model for task execution and auxiliary calls.
    tiers = getattr(cfg.llm.models, provider)
    tiers.best = tiers.coding = tiers.fast = model
    cfg.approval.mode = "standard"
    cfg.filesystem.allow_paths = [str(Path.cwd())]

    from rune.agent.failover import build_profiles_from_config
    from rune.agent.loop import NativeAgentLoop
    from rune.llm.pricing import usage_payload
    from rune.types import AgentConfig

    loop = NativeAgentLoop(AgentConfig(provider=provider, model=model, max_iterations=12,
                                     timeout_seconds=150, _overridden=True))
    tools, approvals, events, routing_decisions = [], [], [], []
    text = []
    first_text = None
    started = time.monotonic()

    async def on_text(delta):
        nonlocal first_text
        if first_text is None and delta:
            first_text = time.monotonic() - started
        text.append(delta)

    async def on_tool(event):
        tools.append(event["name"])
        events.append({"event": "tool_call", **event})

    async def on_result(event):
        events.append({"event": "tool_result", **event})

    async def on_classified(result):
        from dataclasses import asdict
        routing_decisions.append({**asdict(result), "intent_categories": sorted(result.intent_categories)})

    async def deny_approval(capability, reason):
        approvals.append({"capability": capability, "reason": reason})
        return False

    loop.on("text_delta", on_text)
    loop.on("tool_call", on_tool)
    loop.on("tool_result", on_result)
    loop.on("goal_classified", on_classified)
    loop.set_approval_callback(deny_approval)
    workspace = Path.cwd()
    goal = prepare(case, workspace)
    report = {"scenario": case, "provider": provider, "model": model}
    try:
        # Keep fallback models out of comparisons for the selected model.
        with patch("rune.agent.failover.build_profiles_from_config", return_value=build_profiles_from_config()[:1]):
            trace = await asyncio.wait_for(loop.run(goal, context={"workspace_root": str(workspace)}), 170)
        answer = loop._last_answer_text or "".join(text)
        checks = verify(case, workspace, answer)
        if case == "code":
            checks["before_after_recorded"] = verified_code_comparison(loop._verification)
        table_verification = getattr(trace, "table_acceptance", {}) or {}
        if case == "csv":
            checks["table_verified"] = table_verification.get("status") == "pass"
            checks["row_order_verified"] = any(
                "row_order" in result.get("checks", []) and result.get("status") == "pass"
                for result in table_verification.get("results", []))
        checks["no_desktop_or_browser"] = not any(t.startswith(("desktop", "browser")) for t in tools)
        checks["no_approval_needed"] = not approvals
        report.update(reason=trace.reason, answer=answer, checks=checks, usage=usage_payload(trace),
                      timings=trace.timings, completion_check=trace.completion_check,
                      table_verification=table_verification,
                      passed=all(checks.values()) and trace.reason in {"completed", "verified"})
    except Exception as exc:
        timings = getattr(loop, "_last_run_timings", {})
        report.update(passed=False, error=type(exc).__name__, timings=timings,
                      usage=usage_payload(SimpleNamespace(timings=timings)))
    report.update(seconds=round(time.monotonic() - started, 2), first_text_seconds=first_text,
                  tools=tools, approvals=approvals, events=events, routing=routing_decisions)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def comparison_summary(reports: list[dict]) -> list[dict]:
    groups = {}
    for report in reports:
        key = (report["model"], report["scenario"], report["requested_backend"])
        groups.setdefault(key, []).append(report)
    summaries = []
    for (model, scenario, backend), rows in groups.items():
        skipped = sum(bool(r.get("skipped")) for r in rows)
        rows = [r for r in rows if not r.get("skipped")]
        seconds = sorted(r["seconds"] for r in rows if "seconds" in r)
        costs = [(r.get("usage") or {}).get("cost") or {} for r in rows]
        complete = bool(rows) and all(c.get("usd") is not None for c in costs)
        passed = sum(r["passed"] for r in rows)
        summaries.append({"model": model, "scenario": scenario, "backend": backend,
            "attempts": len(rows), "skipped": skipped, "passed": sum(r["passed"] for r in rows),
            "median_seconds": statistics.median(seconds) if seconds and len(seconds) == len(rows) else None,
            "max_seconds": max(seconds) if seconds else None,
            "mean_usd": sum(c["usd"] for c in costs) / len(rows) if complete else None,
            "usd_per_verified_completion": sum(c["usd"] for c in costs) / passed if complete and passed else None,
            "known_usd": sum(c.get("knownUsd", 0) for c in costs),
            "unpriced_attempts": sum(c.get("usd") is None for c in costs)})
    return summaries


def provider_blocker(report: dict) -> str | None:
    if report.get("passed"):
        return None
    errors = [s["error"] for s in (report.get("timings") or {}).get("spans", []) if s.get("error")]
    if sum(e.get("status") == 429 for e in errors) >= 2:
        return "repeated_rate_limit"
    if any(e.get("status") in {401, 402, 403} for e in errors):
        return "provider_access_denied"
    # Stop this provider if the request is rejected before any tool runs.
    if not report.get("tools") and any(e.get("status") == 400 for e in errors):
        return "routing_request_rejected"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rune-core-results.json"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--routing", choices=("connected", "jev", "both"), default="connected")
    parser.add_argument("--provider", choices=tuple(p for p, _ in MODELS))
    parser.add_argument("--repeat", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--worker", nargs=3, metavar=("SCENARIO", "PROVIDER", "MODEL"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.routing == "both":
            parser.error("A worker must use one routing backend")
        asyncio.run(worker(*args.worker, args.output, routing=args.routing))
        return 0

    sys.path.insert(0, str(ROOT))
    from rune.config import get_config
    # Load credentials before switching to the temporary RUNE_HOME.
    get_config()
    reports = []
    for provider, model in MODELS:
        if args.provider and args.provider != provider:
            continue
        blocked = None
        for case in args.scenario or SCENARIOS:
            with tempfile.TemporaryDirectory(prefix=".rune-eval-", dir=ROOT) as work:
                arms = ("connected", "jev") if args.routing == "both" else (args.routing,)
                for repeat in range(args.repeat):
                    # Alternate order to reduce cache bias; keep paths stable and reset data.
                    for backend in arms if repeat % 2 == 0 else reversed(arms):
                        if blocked:
                            report = {"model": model, "provider": provider, "scenario": case, "passed": False,
                                      "requested_backend": backend, "repeat": repeat + 1, "skipped": blocked}
                            reports.append(report)
                            args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
                            continue
                        for child in Path(work).iterdir():
                            if child.is_dir() and not child.is_symlink():
                                shutil.rmtree(child)
                            else:
                                child.unlink()
                        with tempfile.TemporaryDirectory(prefix="rune-core-") as state:
                            report = run_worker(case, provider, model, backend, work, state)
                        report.update(requested_backend=backend, repeat=repeat + 1)
                        blocked = provider_blocker(report)
                        reports.append(report)
                        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
                        print(json.dumps({k: report[k] for k in ("model", "scenario", "requested_backend", "repeat",
                                         "passed", "seconds", "error", "checks", "usage") if k in report}), flush=True)
    print(json.dumps({"summary": comparison_summary(reports)}, ensure_ascii=False), flush=True)
    return 0 if all(r["passed"] for r in reports) else 1


def run_worker(case: str, provider: str, model: str, backend: str, work: str, state: str) -> dict:
    result = Path(state) / "result.json"
    env = {**os.environ, "RUNE_HOME": str(Path(state) / "home"), "RUNE_WORKSPACE": work,
           "RUNE_APPROVAL_MODE": "standard", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    print(f"{case}: {model} ({backend})", flush=True)
    started = time.monotonic()
    with (Path(state) / "worker.log").open("w") as log:
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", case, provider, model,
                            "--output", str(result), "--routing", backend],
                           env=env, cwd=work, stdout=log, stderr=log, timeout=190)
        except subprocess.TimeoutExpired:
            pass
    report = json.loads(result.read_text()) if result.exists() else {
        "scenario": case, "provider": provider, "model": model, "passed": False,
        "error": "WorkerTimeoutOrExit", "seconds": round(time.monotonic() - started, 2)}
    diagnostic = (Path(state) / "worker.log").read_text()[-6000:]
    for name, value in os.environ.items():
        if len(value) >= 16 and any(word in name for word in ("KEY", "TOKEN", "SECRET")):
            diagnostic = diagnostic.replace(value, "[redacted]")
    report["diagnostic"] = diagnostic
    return report


if __name__ == "__main__":
    raise SystemExit(main())
