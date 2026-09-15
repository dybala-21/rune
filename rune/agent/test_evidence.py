"""Keep observed test identities separate from aggregate failure counts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# These are runner output formats, not language-dependent answer rules.
_UNIT_CASE = re.compile(r"([\w.]+) \(([\w.]+)\) \.\.\. (ok|FAIL|ERROR|skipped[^\n]*|expected failure|unexpected success)")
_UNIT_FAILURE = re.compile(r"^(FAIL|ERROR): ([\w.]+) \(([\w.]+)\)(?: (.+))?$", re.M)
_UNIT_TOTAL = re.compile(r"^Ran (\d+) tests? in [^\n]+$", re.M)
_UNIT_COUNTS = re.compile(r"^(?:FAILED|OK)(?: \(([^\n]+)\))?$", re.M)
_PYTEST_CASE = re.compile(r"^([^\n]+::[^\n]+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s+\[.*\])?$", re.M)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LIMIT = 128


@dataclass(frozen=True, slots=True)
class TestCase:
    identity: str
    status: str
    subtest_failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TestReport:
    runner: str
    cases: tuple[TestCase, ...]
    tests_run: int | None
    failure_events: int | None
    complete: bool

    def snapshot(self) -> dict:
        return asdict(self)


def parse_test_report(output: str) -> TestReport | None:
    clipped = len(output) > 200_000
    text = output[:100_000] + "\n" + output[-100_000:] if clipped else output
    text = _ANSI.sub("", text)
    total = list(_UNIT_TOTAL.finditer(text))
    if total:
        statuses: dict[str, str] = {}
        subtests: dict[str, list[str]] = {}
        for item in _UNIT_CASE.finditer(text):
            method, owner, result = item.groups()
            status = "pass" if result == "ok" else "fail" if result in {"FAIL", "ERROR", "unexpected success"} else "skip"
            statuses[f"{owner}.{method}"] = status
        for item in _UNIT_FAILURE.finditer(text):
            _, method, owner, subtest = item.groups()
            identity = f"{owner}.{method}"
            statuses[identity] = "fail"
            if subtest:
                subtests.setdefault(identity, []).append(subtest[:300])
        summaries = list(_UNIT_COUNTS.finditer(text))
        failures = None
        if summaries:
            counts = dict(re.findall(r"(failures|errors|unexpected successes|skipped|expected failures)=(\d+)", summaries[-1].group(1) or ""))
            if counts or summaries[-1].group(0).startswith("OK"):
                failures = sum(int(counts.get(key, 0)) for key in ("failures", "errors", "unexpected successes"))
        count = int(total[-1].group(1))
        cases = tuple(TestCase(key, status, tuple(subtests.get(key, []))) for key, status in list(statuses.items())[:_LIMIT])
        complete = not clipped and len(total) == 1 and len(cases) == count and "characters omitted" not in text
        return TestReport("unittest", cases, count, failures, complete)
    matches = list(_PYTEST_CASE.finditer(text))
    if matches:
        mapping = {"PASSED": "pass", "FAILED": "fail", "ERROR": "fail", "SKIPPED": "skip", "XFAIL": "skip", "XPASS": "unknown"}
        cases = tuple(TestCase(m.group(1)[:500], mapping[m.group(2)]) for m in matches[:_LIMIT])
        # Verbose case lines need not cover collection errors or a truncated summary.
        return TestReport("pytest", cases, None, None, False)
    return None
