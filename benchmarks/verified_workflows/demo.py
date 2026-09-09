"""Render fictional office data through registered capabilities without an LLM.

Run: python -m benchmarks.verified_workflows.demo --output /path/to/demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rune.capabilities.document import (
    DocSheet,
    DocumentCreateParams,
    document_create,
)
from rune.capabilities.document_bundle import DocumentBundleParams, document_bundle

SALES = [
    ["주문", "상태", "금액"],
    ["A001", "확정", 12_000_000],
    ["A002", "확정", 8_000_000],
    ["A003", "대기", 5_000_000],
    ["A004", "취소", 4_000_000],
    ["A005", "확정", 3_000_000],
    ["A006", "환불", 2_000_000],
]


def specification(source: Path, directory: Path, *, confirmed_only: bool) -> DocumentBundleParams:
    scope = "확정 주문만 포함" if confirmed_only else "모든 상태의 주문 포함"
    summary = "집계 금액은 {{amount}}원이며 주문은 {{orders}}건입니다."
    blocks = [
        {"type": "paragraph", "text": summary},
        {"type": "heading", "text": "집계 기준"},
        {"type": "paragraph", "text": scope + ". 단위는 원입니다."},
        {"type": "table", "rows": [["항목", "값"], ["금액", "{{amount}}"], ["주문 수", "{{orders}}"]]},
        {"type": "heading", "text": "회의 안건"},
        {"type": "paragraph", "text": "대기 주문의 확정 가능성을 확인하고 취소 및 환불 사유를 검토합니다."},
        {"type": "paragraph", "text": "출처는 sales.xlsx의 주문 시트입니다."},
    ]
    return DocumentBundleParams.model_validate({
        "directory": str(directory), "source_path": str(source), "sheet": "주문",
        "filters": [{"column": "상태", "operator": "eq", "values": ["확정"]}] if confirmed_only else [],
        "metrics": [{"id": "amount", "operation": "sum", "column": "금액"},
                    {"id": "orders", "operation": "count"}],
        "documents": [
            {"filename": "summary.xlsx", "format": "xlsx", "sheets": [
                {"name": "집계", "rows": [["항목", "값"], ["금액 원", "{{amount}}"],
                                          ["주문 수", "{{orders}}"]]},
                {"name": "기준", "rows": [["항목", "내용"], ["포함 범위", scope],
                                          ["출처", "sales.xlsx 주문 시트"]]},
            ]},
            {"filename": "briefing.docx", "format": "docx", "font_family": "Arial Unicode MS",
             "title": "주문 현황 회의 브리핑", "blocks": blocks},
            {"filename": "briefing.pdf", "format": "pdf", "title": "주문 현황 회의 브리핑", "blocks": blocks},
            {"filename": "meeting.pptx", "format": "pptx", "font_family": "Arial Unicode MS",
             "title": "주문 현황 회의", "blocks": [
                {"type": "heading", "text": "주문 집계"},
                {"type": "bullets", "items": [summary, scope, "출처는 sales.xlsx 주문 시트입니다."]},
                {"type": "heading", "text": "회의 안건"},
                {"type": "bullets", "items": ["대기 주문의 확정 가능성 확인", "취소 및 환불 사유 검토"]},
            ]},
        ],
    })


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = output / "sales.xlsx"
    created = await document_create(DocumentCreateParams(
        path=str(source), format="xlsx", sheets=[DocSheet(name="주문", rows=SALES)],
    ))
    if not created.success:
        raise RuntimeError(created.error)
    reports = []
    for confirmed in (False, True):
        params = specification(source, output / "deliverables", confirmed_only=confirmed)
        result = await document_bundle(params)
        if not result.success:
            raise RuntimeError(result.error)
        expected = {"amount": 23_000_000, "orders": 3} if confirmed else {"amount": 34_000_000, "orders": 6}
        if result.metadata["metrics"] != expected:
            raise AssertionError(f"Independent expected totals differ: {result.metadata}")
        reports.append({"condition": "confirmed_only" if confirmed else "all_orders", **result.metadata})
    report = {"fixture": "fictional Korean orders", "live_model": False, "runs": reports}
    (output / "demo-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.output.resolve())), ensure_ascii=False, indent=2))
