"""Shared test fixtures for RUNE."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


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
