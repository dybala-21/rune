import json
import subprocess

import pytest

from scripts.e2e_provenance import ReportWriter, background_run, model_settings, source_state


def test_source_fingerprint_includes_local_changes_and_excludes_outputs(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    (tmp_path / "rune").mkdir()
    source = tmp_path / "rune" / "example.py"
    source.write_text("value = 1\n")
    git("add", "rune")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-m", "fixture")
    original = source_state(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "report.json").write_text("{}")
    assert source_state(tmp_path) == original
    source.write_text("value = 2\n")
    edited = source_state(tmp_path)
    assert edited["commit"] == original["commit"] and edited["content_hash"] != original["content_hash"]
    (tmp_path / "rune" / "new.py").write_text("enabled = True\n")
    assert source_state(tmp_path)["content_hash"] != edited["content_hash"]


def test_report_writer_never_overwrites_and_marks_mid_trial_changes(tmp_path):
    source = {"commit": "abc", "content_hash": "tree"}
    def writer():
        return ReportWriter(tmp_path, source=source, batch_id="batch", scenario="test", scenario_hash="case", environment={})

    first, second = writer(), writer()
    path = first.write({"outcome": "passed"}, settings={}, current_source=source)
    other = second.write({"outcome": "failed"}, settings={}, current_source={**source, "content_hash": "changed"})
    assert path != other
    assert json.loads(path.read_text())["evaluation"]["source_unchanged"] is True
    assert json.loads(other.read_text())["evaluation"]["source_unchanged"] is False
    with pytest.raises(ValueError, match="already has a report"):
        first.write({}, settings={})


def test_metadata_excludes_credentials():
    from rune.config.schema import RuneConfig

    cfg = RuneConfig(openai_api_key="secret-value")
    assert "secret-value" not in json.dumps(model_settings(cfg))


def test_background_usage_requires_complete_billing():
    result = {"duration_ms": 1500, "timings": {"usage": {
        "calls": 2, "reported_calls": 1, "cost_usd": .3, "total_tokens": 50,
    }}}
    normalized = background_run(result)
    assert normalized["seconds"] == 1.5
    assert normalized["snapshot"]["usage"]["cost"] == {"usd": None, "knownUsd": .3}
