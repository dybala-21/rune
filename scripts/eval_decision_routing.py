#!/usr/bin/env python3
"""Compare routing accuracy, fallback rate, latency and cost on synthetic requests.

Usage: .venv/bin/python scripts/eval_decision_routing.py --output /tmp/routing.json
Credentials come from Rune's existing settings. No apps or task tools are used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = (("xai", "grok-4.6"), ("gemini", "gemini-2.5-flash"), ("anthropic", "claude-opus-5"))

# Score only explicit requirements; a CSV request can legitimately require a script.
CASES = (
    ("chat", "ESTABLISHED TCP 소켓이 뭔지 설명해줘.", {"desktop": False, "requires_execution": False}),
    ("csv", "앱을 열지 말고 expenses.csv를 부서별로 합산해서 summary.csv로 저장해줘.", {"desktop": False, "table": True}),
    ("xlsx", "sales.csv에서 지역별 매출 합계를 계산해서 result.xlsx 파일로 만들어줘.", {"desktop": False, "table": True}),
    ("template", "입력 데이터 없이 월별 지출을 기록할 빈 Excel 템플릿을 만들어줘.", {"desktop": False, "table": False}),
    ("read_app", "TextEdit에 열려 있는 미저장 문서를 읽고 요약해줘.", {"desktop": True, "requires_desktop_input": False}),
    ("write_app", "TextEdit에서 현재 열려 있는 문서의 제목을 수정하고 저장해줘.", {"desktop": True, "requires_desktop_input": True}),
    ("calculator", "Calculator 앱으로 173 곱하기 29를 계산해줘.", {"desktop": True, "requires_desktop_input": True, "requires_execution": False, "calculation_expression": ""}),
    ("browser", "브라우저에서 영화 좌석 선택 화면까지 진행하고 결제 전에 멈춰줘.", {"goal_type": "browser", "desktop": False}),
    ("code", "stats.py를 수정하고 기존 테스트를 실행해. 수정 전후 결과를 Markdown 표로 알려줘.", {"desktop": False, "requires_execution": True, "table": False}),
    ("review", "저장소 코드를 읽고 개선할 점만 설명해. 코드를 수정하거나 실행하지 마.", {"desktop": False, "requires_execution": False}),
    ("arithmetic", "173 × 29 − 417의 값을 숫자로만 답해줘.", {"desktop": False, "calculation_expression": "173 × 29 − 417"}),
    ("decimal", "Evaluate 19.95 × 7 - 12.4. Reply with just the numeric result.", {"desktop": False, "calculation_expression": "19.95 × 7 - 12.4"}),
    ("english", "Read the unsaved document open in TextEdit. Do not edit it.", {"desktop": True, "requires_desktop_input": False}),
    ("negation", "Do not use Excel. Aggregate data.csv into totals.csv using file tools.", {"desktop": False, "table": True}),
    ("email", "아래 이메일 초안을 더 정중하게 고쳐줘: 내일까지 답 주세요.", {"desktop": False, "email": True}),
    ("web", "https://example.org 페이지 내용을 읽어서 요약해줘.", {"goal_type": "web", "desktop": False}),
    ("mixed", "열린 TextEdit 문서를 수정하고 저장한 뒤 프로젝트 테스트도 실행해줘.", {"desktop": True, "requires_desktop_input": True, "requires_execution": True}),
    ("identifier", "release-2026-09 버전명의 의미를 설명해줘.", {"desktop": False, "calculation_expression": ""}),
)
HELD_OUT = (
    ("not_app", "Numbers로 열 수 있게 원본 매출 데이터를 분기별 합계 XLSX로 만들어줘. 앱 실행은 필요 없어.", {"desktop": False, "table": True}),
    ("app_read", "지금 Numbers 창에 표시된 합계만 확인하고 말해줘. 셀은 건드리지 마.", {"desktop": True, "requires_desktop_input": False}),
    ("app_write", "지금 Numbers에 열려 있는 표에서 B2 값을 120으로 바꿔줘.", {"desktop": True, "requires_desktop_input": True}),
    ("no_code_run", "테스트 코드를 예시로만 보여줘. 파일 저장이나 실행은 하지 마.", {"desktop": False, "requires_execution": False}),
    ("spreadsheet_code", "CSV를 XLSX로 변환하는 Python 코드를 작성하고 테스트해줘.", {"desktop": False, "requires_execution": True, "table": False}),
    ("draft", "앱을 열지 말고 아래 메모로 보고서 본문만 작성해줘: 출시 일정은 다음 주.", {"desktop": False, "requires_execution": False}),
    ("browser_local", "http://localhost:3000 웹페이지의 로그인 버튼을 눌러줘.", {"goal_type": "browser", "desktop": False}),
    ("negative_arithmetic", "Compute (-2 + 7) × 3 exactly.", {"desktop": False, "calculation_expression": "(-2 + 7) × 3"}),
    ("quoted_arithmetic", "Do not calculate 19.95 * 7. Explain what the asterisk means in Python.", {"desktop": False, "calculation_expression": ""}),
    ("japanese", "TextEditで開いている文書を読んで要約してください。変更しないでください。", {"desktop": True, "requires_desktop_input": False}),
    ("unrelated", "이제 TCP 연결의 ESTABLISHED 상태를 설명해줘.", {"desktop": False, "is_domain_change": True}),
    ("followup", "제목만 바꾸고 저장해줘.", {"desktop": True, "requires_desktop_input": True, "is_domain_change": False}),
)


async def run(args) -> bool:
    sys.path.insert(0, str(ROOT))
    from rune.agent.goal_classifier import classify_goal
    from rune.agent.timing import capture_timing, timing_snapshot
    from rune.config import get_config
    from rune.utils.logger import configure_logging

    configure_logging(level="ERROR")
    cfg = get_config()
    rows = []
    cases = CASES if args.suite == "calibration" else HELD_OUT
    for provider, model in MODELS:
        if args.provider and provider != args.provider:
            continue
        cfg.llm.active_provider, cfg.llm.active_model = provider, model
        for backend in args.backend or ("connected", "jev"):
            cfg.llm.decision_routing.backend = backend
            semaphore = asyncio.Semaphore(args.concurrency)

            async def one(case, semaphore=semaphore, model=model, backend=backend):
                async with semaphore:
                    name, goal, expected = case
                    previous = "TextEdit에서 열려 있는 문서를 수정해줘." if name in {"unrelated", "followup"} else ""
                    started = time.monotonic()
                    with capture_timing() as run:
                        result = await classify_goal(goal, previous_goal=previous, previous_goal_type="full" if previous else "")
                        timings = timing_snapshot(run)
                    actual = {key: (key in result.intent_categories if key in {"desktop", "table", "email"}
                                    else getattr(result, key)) for key in expected}
                    row = {"case": name, "model": model, "requested_backend": backend,
                           "backend": result.decision_backend, "fallback": result.fallback_reason,
                           "passed": result.available and actual == expected,
                           "actual": actual, "expected": expected,
                           "seconds": round(time.monotonic() - started, 3), "timings": timings}
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in ("case", "model", "backend", "fallback", "passed", "seconds")}), flush=True)

            await asyncio.gather(*(one(case) for case in cases))
            args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            arm = [r for r in rows if r["model"] == model and r["requested_backend"] == backend]
            durations = sorted(r["seconds"] for r in arm)
            print(json.dumps({"model": model, "backend": backend, "passed": sum(r["passed"] for r in arm),
                              "total": len(arm), "median_seconds": statistics.median(durations),
                              "fallbacks": sum(bool(r["fallback"]) for r in arm),
                              "known_usd": sum(r["timings"]["usage"]["cost_usd"] for r in arm)}), flush=True)
    return all(r["passed"] for r in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("connected", "jev"), action="append")
    parser.add_argument("--provider", choices=tuple(provider for provider, _ in MODELS))
    parser.add_argument("--suite", choices=("calibration", "held-out"), default="held-out")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rune-decision-routing.json"))
    return 0 if asyncio.run(run(parser.parse_args())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
