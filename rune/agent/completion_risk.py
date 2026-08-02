"""Estimate whether a finished run actually did what it says it did.

Off the paths where something can be executed, a run that failed and a run
that succeeded end the same way: a confident summary. Asking a model to
read that summary and judge it barely beats a coin toss, because the tell
it keys on — assured phrasing — is present either way.

What separates them is the shape of the work. A run that quietly gave up
reads the same things over and over and stops; a run that got somewhere
writes, checks, and writes again. So the features here are drawn from the
sequence of tool calls and nothing else. The final message is deliberately
excluded: rewriting a closing paragraph in a more confident register flips
detectors that look at it, and changes nothing about what was done.

The model is a small logistic regression over tool-name n-grams, trained
offline from labelled runs and loaded from a weights file. There is no
default model shipped — without weights this reports nothing rather than
guessing, since a miscalibrated warning on honest work is worse than
silence.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_data

log = get_logger(__name__)

_ENV_FLAG = "RUNE_COMPLETION_RISK"
_ENV_MODEL = "RUNE_COMPLETION_RISK_MODEL"
_MAX_CALLS = 400


def risk_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def tool_sequence(calls: list[str]) -> list[str]:
    """Normalise a run's tool names into the sequence used for features."""
    return [c.strip() for c in calls[:_MAX_CALLS] if c and c.strip()]


def features(calls: list[str]) -> dict[str, float]:
    """Counts over the tool-call sequence — no natural language anywhere.

    Unigrams and bigrams capture what was done and in what order; the
    summary terms capture the shapes that separate the two outcomes in
    practice: repeated reading with nothing written, versus writing
    followed by a check and then more writing.
    """
    seq = tool_sequence(calls)
    f: dict[str, float] = {}
    if not seq:
        return {"bias": 1.0, "empty": 1.0}
    for name in seq:
        f[f"t:{name}"] = f.get(f"t:{name}", 0.0) + 1.0
    for a, b in zip(seq, seq[1:], strict=False):
        f[f"b:{a}>{b}"] = f.get(f"b:{a}>{b}", 0.0) + 1.0

    reads = sum(1 for c in seq if c.startswith(("file_read", "file_list",
                                                "file_search", "code_")))
    writes = sum(1 for c in seq if c.startswith(("file_write", "file_edit")))
    shells = sum(1 for c in seq if c.startswith("bash"))
    n = float(len(seq))
    f["bias"] = 1.0
    f["n_calls"] = n / 50.0
    f["read_frac"] = reads / n
    f["write_frac"] = writes / n
    f["shell_frac"] = shells / n
    f["no_writes"] = 1.0 if writes == 0 else 0.0
    # A write that is followed by something being run, then written again,
    # is the signature of work that was checked rather than assumed.
    f["write_check_write"] = 0.0
    for i in range(len(seq) - 2):
        if (seq[i].startswith(("file_write", "file_edit"))
                and seq[i + 1].startswith("bash")
                and any(s.startswith(("file_write", "file_edit"))
                        for s in seq[i + 2:])):
            f["write_check_write"] = 1.0
            break
    # Reading the same thing again and again is the giving-up shape.
    repeats = len(seq) - len(set(seq))
    f["repeat_frac"] = repeats / n
    return f


@dataclass
class RiskModel:
    weights: dict[str, float]
    threshold: float = 0.5

    def score(self, calls: list[str]) -> float:
        """Probability that a completion claim from this run is wrong."""
        z = sum(self.weights.get(k, 0.0) * v for k, v in features(calls).items())
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    @classmethod
    def load(cls, path: str | Path | None = None) -> RiskModel | None:
        p = Path(path or os.environ.get(_ENV_MODEL, "")
                 or Path(rune_data()) / "completion_risk.json")
        try:
            blob = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        w = blob.get("weights")
        if not isinstance(w, dict) or not w:
            return None
        return cls(weights={str(k): float(v) for k, v in w.items()},
                   threshold=float(blob.get("threshold", 0.5)))


def train(samples: list[tuple[list[str], int]], *, epochs: int = 400,
          lr: float = 0.5, l2: float = 1e-3) -> dict[str, float]:
    """Fit weights from (tool-call sequence, label) pairs. 1 = claim was wrong.

    Plain batch gradient descent so this carries no dependency; the feature
    count is small and the data set is a few hundred runs.
    """
    rows = [(features(calls), y) for calls, y in samples]
    keys = sorted({k for f, _ in rows for k in f})
    w = dict.fromkeys(keys, 0.0)
    n = max(1, len(rows))
    for _ in range(epochs):
        grad = dict.fromkeys(keys, 0.0)
        for f, y in rows:
            z = sum(w[k] * v for k, v in f.items() if k in w)
            p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
            err = p - y
            for k, v in f.items():
                if k in grad:
                    grad[k] += err * v
        for k in keys:
            w[k] -= lr * (grad[k] / n + l2 * w[k])
    return w


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC; 0.5 means the score carries no information."""
    pairs = sorted(zip(scores, labels, strict=True))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return 0.5
    rank = 0.0
    i = 0
    ranks: list[float] = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank = sum(r for r, (_, y) in zip(ranks, pairs, strict=True) if y == 1)
    return (rank - pos * (pos + 1) / 2.0) / (pos * neg)


_TOOL_LINE = re.compile(r"\b(file_read|file_write|file_edit|file_list|"
                        r"file_search|file_delete|bash_execute|code_\w+|"
                        r"web_search|web_fetch|task_blocked|think)\b")


def calls_from_log(text: str) -> list[str]:
    """Tool names in the order a run's log shows them being invoked."""
    return _TOOL_LINE.findall(text or "")[:_MAX_CALLS]
