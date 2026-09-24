"""Build and match test tables from recorded runs."""

from __future__ import annotations

import re


def inline_text(tokens) -> str | None:
    """Return plain text, allowing emphasis and inline code."""
    if any(token.type not in {"text", "code_inline", "strong_open", "strong_close", "em_open", "em_close"}
           for token in tokens or []):
        return None
    return "".join(token.content for token in tokens or []).strip()


def recorded_tables(evidence: list[dict], language: str = "en") -> list[str]:
    labels = (("테스트 결과", "수정 전", "수정 후", "전체 실행 결과", "실행 테스트 수", "실패 테스트 수", "실패 발생 수")
              if language == "ko" else
              ("Test results", "Before", "After", "Suite status", "Tests run", "Failed tests", "Failure events"))
    tables = []
    if len(evidence) > 3:
        return tables
    for pair in evidence:
        before, after = pair["before"], pair["after"]
        if not all(r["complete"] and r["failed_tests"] is not None for r in (before, after)):
            continue
        cases = [{c["identity"]: c["status"] for c in report["cases"]} for report in (before, after)]
        names = sorted(cases[0].keys() | cases[1].keys())
        command, cwd = pair["command"], pair.get("cwd", "")
        # Skip values that could change the table structure.
        if (len(names) > 12 or not cwd or len(command) >= 600 or len(cwd) > 400
                or any(not re.fullmatch(r"[\w.:-]+", name) for name in names)
                or any(re.search(r"[\x00-\x1f\x7f`|<>\\]", text) for text in [command, cwd, *names])):
            continue
        rows = [f"| {labels[0]}: `{command}` — `{cwd}` | {labels[1]} | {labels[2]} |",
                "| --- | --- | --- |"]
        for label, key in zip(labels[3:], ("check_status", "tests_run", "failed_tests", "failure_events"), strict=True):
            values = [str(r[key]) if r[key] is not None else "unknown" for r in (before, after)]
            rows.append(f"| {label} | {values[0]} | {values[1]} |")
        rows.extend(f"| `{name}` | {cases[0].get(name, 'unknown')} | {cases[1].get(name, 'unknown')} |" for name in names)
        table = "\n".join(rows)
        if len(table) <= 4000:
            tables.append(table)
    return tables if sum(map(len, tables)) <= 6000 else []


def _table_blocks(text: str, parser):
    tokens = parser.parse(text)
    for index, token in enumerate(tokens):
        if token.type != "table_open" or token.level != 0:
            continue
        rows, row = [], []
        for child in tokens[index + 1:]:
            if child.type == "table_close":
                break
            if child.type == "tr_open":
                row = []
            elif child.type == "inline":
                row.append(inline_text(child.children))
            elif child.type == "tr_close":
                rows.append(row)
        if rows and all(len(row) == 3 and None not in row for row in rows):
            yield token.map, rows


def without_recorded_tables(answer: str, evidence: list[dict], *, single_check: bool = False) -> str:
    """Blank matching tables to preserve source line numbers."""
    if "|" not in answer or len(answer) > 40000:
        return answer
    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark").enable("table")
    expected = []
    for language in ("en", "ko"):
        for table in recorded_tables(evidence, language):
            for _, rows in _table_blocks(table, parser):
                expected.append(rows)
                # With only one check, the directory is optional.
                if single_check and len(evidence) == 1:
                    header = rows[0][0].removesuffix(" — " + evidence[0]["cwd"])
                    expected.append([[header, *rows[0][1:]], *rows[1:]])
    if not expected:
        return answer
    lines = answer.splitlines(keepends=True)
    for (start, end), rows in _table_blocks(answer, parser):
        if rows in expected:
            for line in range(start, end):
                lines[line] = "\n" if lines[line].endswith("\n") else ""
    return "".join(lines)
