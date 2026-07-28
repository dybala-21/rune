"""Rung-0 fast path: parsing, applying, repro gating, best_of wiring."""

from __future__ import annotations

import pytest

import rune.agent.fastpath as fp
from rune.agent.fastpath import (
    _extract_files,
    _repo_tree,
    _skeleton,
    apply_candidate,
    parse_candidate,
    run_fastpath,
)

REPLY = """Looking at the bug, the fix is:

### FILE: pkg/mod.py
```
<<<<<<< SEARCH
def add(a, b):
    return a - b
=======
def add(a, b):
    return a + b
>>>>>>> REPLACE
```
"""


def test_parse_candidate_extracts_file_edits():
    edits = parse_candidate(REPLY)
    assert edits == [
        ("pkg/mod.py", "def add(a, b):\n    return a - b",
         "def add(a, b):\n    return a + b"),
    ]


def test_parse_candidate_ignores_untagged_blocks():
    assert parse_candidate("no file tags here\n```\nx\n```") == []


def _mk_repo(tmp_path, name):
    root = tmp_path / name
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "pkg" / "tests").mkdir()
    (root / "pkg" / "tests" / "test_mod.py").write_text(
        "import sys, os\nsys.path.insert(0, os.getcwd())\n"
        "from pkg.mod import add\n\n"
        "def test_identity():\n    assert add(0, 0) == 0\n"
    )
    (root / "setup.py").write_text("# marker\n")
    return root


def test_apply_candidate_all_or_nothing(tmp_path):
    repo = _mk_repo(tmp_path, "r1")
    edits = parse_candidate(REPLY) + [("pkg/mod.py", "NOT PRESENT", "x")]
    assert apply_candidate(str(repo), edits) == []
    # first edit reverted
    assert "a - b" in (repo / "pkg" / "mod.py").read_text()


def test_apply_candidate_applies_fuzzy(tmp_path):
    repo = _mk_repo(tmp_path, "r2")
    touched = apply_candidate(str(repo), parse_candidate(REPLY))
    assert touched == ["pkg/mod.py"]
    assert "a + b" in (repo / "pkg" / "mod.py").read_text()


def test_repo_tree_and_extract(tmp_path):
    repo = _mk_repo(tmp_path, "r3")
    tree = _repo_tree(str(repo))
    assert "pkg/mod.py" in tree
    picked = _extract_files("I think pkg/mod.py and nothing/else.py", tree)
    assert picked == ["pkg/mod.py"]


def test_skeleton_strips_bodies():
    sk = _skeleton("class A:\n    x = 1\n    def f(self, y):\n        return y*2\n")
    assert "def f(self, y): ..." in sk
    assert "y*2" not in sk


@pytest.mark.asyncio
async def test_run_fastpath_verified_end_to_end(tmp_path, monkeypatch):
    """Discriminating repro + correct candidate → verified, files edited."""
    seed = _mk_repo(tmp_path, "seed")
    work = _mk_repo(tmp_path, "work")

    repro = (
        "```python\nimport sys, os\nsys.path.insert(0, os.getcwd())\n"
        "from pkg.mod import add\nassert add(2, 3) == 5\n```"
    )
    calls = {"n": 0}

    async def fake_complete(model, provider, prompt, n=1):
        calls["n"] += 1
        if "Which files" in prompt:
            return ["pkg/mod.py"]
        if "REPRODUCES" in prompt:
            return [repro]
        return [REPLY] * n

    monkeypatch.setattr(fp, "_complete", fake_complete)

    res = await run_fastpath("add() subtracts instead of adding",
                             str(seed), str(work), None, None)
    assert res.verified
    assert res.applied == ["pkg/mod.py"]
    assert "a + b" in (work / "pkg" / "mod.py").read_text()
    assert "repro" in res.method


@pytest.mark.asyncio
async def test_run_fastpath_rejects_non_discriminating_repro(tmp_path, monkeypatch):
    """A repro that PASSES the broken baseline proves nothing → no verify,
    no evidence handoff."""
    seed = _mk_repo(tmp_path, "seed2")
    work = _mk_repo(tmp_path, "work2")

    passing_repro = "```python\nprint('looks fine')\n```"

    async def fake_complete(model, provider, prompt, n=1):
        if "Which files" in prompt:
            return ["pkg/mod.py"]
        if "REPRODUCES" in prompt:
            return [passing_repro]
        raise AssertionError("must not reach edit generation")

    monkeypatch.setattr(fp, "_complete", fake_complete)

    res = await run_fastpath("bug", str(seed), str(work), None, None)
    assert not res.verified
    assert res.repro_script == ""


@pytest.mark.asyncio
async def test_run_fastpath_wrong_candidate_reverts(tmp_path, monkeypatch):
    """Candidate that doesn't flip the repro is reverted; evidence kept."""
    seed = _mk_repo(tmp_path, "seed3")
    work = _mk_repo(tmp_path, "work3")

    repro = (
        "```python\nimport sys, os\nsys.path.insert(0, os.getcwd())\n"
        "from pkg.mod import add\nassert add(2, 3) == 5\n```"
    )
    wrong = REPLY.replace("return a + b", "return a * b")

    async def fake_complete(model, provider, prompt, n=1):
        if "Which files" in prompt:
            return ["pkg/mod.py"]
        if "REPRODUCES" in prompt:
            return [repro]
        return [wrong] * n

    monkeypatch.setattr(fp, "_complete", fake_complete)

    res = await run_fastpath("bug", str(seed), str(work), None, None)
    assert not res.verified
    assert res.repro_script  # evidence for the agentic rung
    assert "a - b" in (work / "pkg" / "mod.py").read_text()  # reverted
