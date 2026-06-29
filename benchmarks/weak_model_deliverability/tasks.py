"""Multi-step coding tasks with HIDDEN ground-truth verifiers.

Each task asks an agent to produce a small module + tests and run them. Scoring
ignores the agent's own tests and checks correctness traps directly (population
vs sample stdev, even-length median, tie-breaks, edge cases), so an agent can't
pass by writing weak tests. The point of the benchmark is deliverability on weak
local models: does the harness make the model produce a VERIFIABLE artifact at
all (vs nothing), and how correct is it.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

CheckResult = tuple[str, bool, str]  # (name, passed, detail)


def _load(workspace: Path, module: str) -> Any | None:
    """Import ``<workspace>/<module>.py`` in isolation, or None if missing/broken."""
    path = workspace / f"{module}.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_wmd_{module}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception:
        return None


def _verify_stats(workspace: Path) -> list[CheckResult]:
    m = _load(workspace, "stats")
    if m is None or not hasattr(m, "summary"):
        return [("module_importable", False, "stats.py / summary() missing or broken")]
    s = m.summary
    out: list[CheckResult] = []

    def chk(name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:  # a trap that raises is a fail, not a crash
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        out.append((name, bool(ok), str(detail)))

    chk(
        "median_even",
        lambda: (abs(s([1, 2, 3, 4])["median"] - 2.5) < 1e-9, s([1, 2, 3, 4])["median"]),
    )
    chk("mode_tie_smallest", lambda: (s([3, 1, 1, 3, 2])["mode"] == 1, s([3, 1, 1, 3, 2])["mode"]))
    chk(
        "population_stdev",
        lambda: (
            abs(s([1, 2, 3, 4, 5])["stdev"] - math.sqrt(2)) < 1e-6,
            s([1, 2, 3, 4, 5])["stdev"],
        ),
    )
    chk("single_element", lambda: (s([5])["stdev"] == 0 and s([5])["median"] == 5, s([5])))

    def empty_raises() -> tuple[bool, Any]:
        try:
            s([])
            return False, "no error"
        except ValueError:
            return True, "ValueError"
        except Exception as exc:
            return False, f"wrong exc {type(exc).__name__}"

    chk("empty_raises_valueerror", empty_raises)
    chk(
        "basics",
        lambda: (s([2, 4, 6])["count"] == 3 and abs(s([2, 4, 6])["mean"] - 4) < 1e-9, "ok"),
    )
    return out


def _verify_wordfreq(workspace: Path) -> list[CheckResult]:
    m = _load(workspace, "wordfreq")
    if m is None or not hasattr(m, "top_words"):
        return [("module_importable", False, "wordfreq.py / top_words() missing or broken")]
    f = m.top_words
    out: list[CheckResult] = []

    def chk(name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        out.append((name, bool(ok), str(detail)))

    chk(
        "case_insensitive",
        lambda: (f("The the THE cat", 1) == [("the", 3)], f("The the THE cat", 1)),
    )
    chk(
        "strips_punctuation",
        lambda: (f("cat, cat. dog!", 1) == [("cat", 2)], f("cat, cat. dog!", 1)),
    )
    chk(
        "tie_broken_alphabetically",
        lambda: (
            [w for w, _ in f("banana apple cherry", 3)] == ["apple", "banana", "cherry"],
            f("banana apple cherry", 3),
        ),
    )
    chk("n_larger_than_vocab", lambda: (len(f("a b a", 99)) == 2, f("a b a", 99)))
    chk("empty_text", lambda: (f("", 5) == [], f("", 5)))
    return out


# Public task registry. Each task: id, instruction (model-facing), verify(workspace).
TASKS: list[dict[str, Any]] = [
    {
        "id": "stats",
        "module": "stats.py",
        "instruction": (
            "Create a Python module stats.py with a function summary(numbers) that returns "
            "a dict with keys: count, mean, median, mode, stdev, min, max. Rules: median of an "
            "even-length list is the average of the two middle values; mode is the most frequent "
            "value and on a tie the SMALLEST such value; stdev must be the POPULATION standard "
            "deviation (not sample); raise ValueError if numbers is empty. Then write test_stats.py "
            "with pytest tests covering these rules and edge cases, run pytest, and fix any failures "
            "so all tests pass. Work only in the current directory."
        ),
        "verify": _verify_stats,
        "max": 6,
    },
    {
        "id": "wordfreq",
        "module": "wordfreq.py",
        "instruction": (
            "Create a Python module wordfreq.py with a function top_words(text, n) that returns the "
            "n most frequent words in text as a list of (word, count) tuples, highest count first. "
            "Rules: case-insensitive (lowercase the words); strip leading/trailing punctuation so "
            "'cat,' counts as 'cat'; on a count tie, order tied words alphabetically; if n is larger "
            "than the vocabulary, return all words; return [] for empty text. Then write "
            "test_wordfreq.py with pytest tests for these rules and edge cases, run pytest, and fix "
            "any failures so all tests pass. Work only in the current directory."
        ),
        "verify": _verify_wordfreq,
        "max": 5,
    },
]
