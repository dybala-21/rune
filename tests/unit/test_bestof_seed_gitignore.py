"""Seeding honors the project's own .gitignore instead of a hardcoded list.

In a git work tree, what gets copied into best-of attempts is what the project
declares as source (tracked + untracked-unignored). The static pattern list
only applies outside git — so new ecosystems (cargo target/, gradle caches,
.next, …) never need entries here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rune.cli.best_of import _seed_file_list, _seed_footprint, _seed_workdir


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


class TestGitAwareSeeding:
    def test_gitignored_build_dir_excluded_without_hardcoding(self, tmp_path):
        root = _git_repo(tmp_path)
        (root / ".gitignore").write_text("weird-build-dir/\n")
        (root / "src.py").write_text("print(1)")
        big = root / "weird-build-dir"
        big.mkdir()
        (big / "artifact.bin").write_bytes(b"x" * 10_000)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)

        listed = _seed_file_list(str(root))
        assert listed is not None
        assert "src.py" in listed and ".gitignore" in listed
        assert not any("weird-build-dir" in p for p in listed)

        files, total = _seed_footprint(str(root))
        assert total < 10_000  # the artifact never counts

        dest = tmp_path / "attempt"
        _seed_workdir(str(root), str(dest))
        assert (dest / "src.py").exists()
        assert not (dest / "weird-build-dir").exists()
        assert not (dest / ".git").exists()

    def test_untracked_unignored_files_are_included(self, tmp_path):
        root = _git_repo(tmp_path)
        (root / "tracked.py").write_text("a")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        (root / "new_untracked.py").write_text("b")  # user's fresh WIP file

        listed = _seed_file_list(str(root))
        assert "new_untracked.py" in listed

        dest = tmp_path / "attempt"
        _seed_workdir(str(root), str(dest))
        assert (dest / "new_untracked.py").exists()

    def test_non_git_dir_falls_back_to_patterns(self, tmp_path):
        root = tmp_path / "plain"
        (root / "node_modules").mkdir(parents=True)
        (root / "node_modules" / "big.js").write_text("x" * 5000)
        (root / "app.js").write_text("ok")

        assert _seed_file_list(str(root)) is None
        files, total = _seed_footprint(str(root))
        assert total < 5000  # node_modules excluded by the fallback list

        dest = tmp_path / "attempt2"
        _seed_workdir(str(root), str(dest))
        assert (dest / "app.js").exists()
        assert not (dest / "node_modules").exists()
