"""Identify the code and settings used by a live evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("rune", "tests", "scripts", "web", "extension", "pyproject.toml", "uv.lock")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_state(root: Path = ROOT) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SOURCE_PATHS)
    entries = []
    for name in sorted(set(names.decode().split("\0")) - {""}):
        path = root / name
        value = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        entries.append((name, value))
    return {"commit": git("rev-parse", "HEAD").decode().strip(), "content_hash": digest(entries)}


def model_settings(cfg) -> dict:
    # Omit credentials, endpoints and user paths from reports.
    return {"reasoning_effort": cfg.llm.reasoning_effort,
            "reasoning_efforts": dict(cfg.llm.reasoning_efforts),
            "route_simple_queries": cfg.llm.route_simple_queries,
            "decision_routing": cfg.llm.decision_routing.model_dump(mode="json")}


def background_run(result: dict) -> dict:
    timing = result.get("timings") or {}
    usage = timing.get("usage") or {}
    complete = bool(usage.get("calls")) and usage.get("calls") == usage.get("reported_calls") and not usage.get("unpriced_calls")
    return {"seconds": result.get("duration_ms", 0) / 1000,
            "snapshot": {"timings": timing, "usage": {
                "total": usage.get("total_tokens"), "cacheRead": usage.get("cached_input_tokens"),
                "cacheCreation": usage.get("cache_write_tokens"),
                "cost": {"usd": usage.get("cost_usd") if complete else None,
                         "knownUsd": usage.get("cost_usd")},
            }}}


class ReportWriter:
    def __init__(self, directory: Path, *, source: dict, batch_id: str, scenario: str,
                 scenario_hash: str, environment: dict):
        self.directory = directory
        self.source = source
        self.batch_id = batch_id
        self.trial_id = uuid4().hex
        self.scenario = scenario
        self.scenario_hash = scenario_hash
        self.environment = environment
        self.written = False

    def write(self, report: dict, *, settings: dict, current_source: dict | None = None) -> Path:
        if self.written:
            raise ValueError("This trial already has a report")
        report = {**report, "case": self.scenario, "evaluation": {
            "version": 1, "batch_id": self.batch_id, "trial_id": self.trial_id,
            "source": self.source, "source_unchanged": (current_source or source_state()) == self.source,
            "scenario_hash": self.scenario_hash, "settings": settings,
            "environment": self.environment,
        }}
        target = self.directory / self.batch_id / f"{self.trial_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        self.written = True
        return target


def environment() -> dict:
    packages = {}
    for name in ("litellm", "pytest", "playwright", "openpyxl", "python-docx"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages}
