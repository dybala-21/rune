"""Shared test fixtures for RUNE."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("live models")
    group.addoption("--run-live", action="store_true", help="Allow E2E tests to call paid model APIs")
    group.addoption("--live-provider", help="Provider for this E2E run")
    group.addoption("--live-model", help="Model ID for this E2E run")
    group.addoption("--live-report-dir", help="Directory for workflow results")
    group.addoption("--decision-backend", choices=("connected", "jev"), help="Override task routing for this test run")


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_home(tmp_dir, monkeypatch):
    """Override HOME to a temp directory."""
    monkeypatch.setenv("HOME", str(tmp_dir))
    return tmp_dir


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons between tests."""
    yield
    # Reset config
    from rune.config.loader import reset_config
    reset_config()
    # Reset prediction engine (prevents real tool_call_log data from leaking into tests)
    import rune.proactive.prediction.engine as pe_mod
    pe_mod._engine = None
    # Reset proactive engine
    import rune.proactive.engine as pro_mod
    pro_mod._engine = None
    # Reset memory store (prevents real DB data from seeding into prediction engine)
    import rune.memory.store as store_mod
    store_mod._store = None
    # Reset skill registry
    import rune.skills.registry as skill_mod
    skill_mod._registry = None
