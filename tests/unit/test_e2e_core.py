import subprocess
import sys

import pytest

from rune.agent.verification_state import VerificationState
from scripts.e2e_core import prepare, verified_code_comparison


@pytest.mark.parametrize("change", ["extra", "missing", "replaced"])
def test_code_comparison_tracks_original_cases_when_the_suite_changes(tmp_path, change):
    prepare("code", tmp_path)
    state = VerificationState()

    def check():
        result = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=tmp_path,
                                capture_output=True, text=True, timeout=10)
        state.observe_command("python3 -m unittest -v", result.returncode == 0,
                              result.stdout + result.stderr, str(tmp_path))

    check()
    state.changed()
    (tmp_path / "stats.py").write_text(
        "def average(values):\n    if not values: raise ValueError('empty')\n    return sum(values) / len(values)\n")
    tests = tmp_path / "test_stats.py"
    if change == "extra":
        (tmp_path / "test_extra.py").write_text(
            "import unittest\nfrom stats import average\nclass Extra(unittest.TestCase):\n"
            "    def test_fraction(self): self.assertEqual(average([1, 2]), 1.5)\n")
    elif change == "missing":
        tests.write_text(tests.read_text().replace("def test_single", "def helper_single"))
    else:
        tests.write_text(tests.read_text().replace("class AverageTests", "class UnrelatedTests"))
    check()
    assert state.passed
    assert verified_code_comparison(state) is (change == "extra")
