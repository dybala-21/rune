"""Shared fixtures for E2E tests that hit real LLM APIs."""

from __future__ import annotations

import os

import pytest

# Skip entire module if no API key is available
_has_api_key = bool(
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip e2e tests when no API key is set."""
    if _has_api_key:
        return
    skip_marker = pytest.mark.skip(reason="No LLM API key — skipping e2e tests")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv():
    """Load ~/.rune/.env so API keys are available."""
    from rune.config.loader import _load_dotenv as load_env
    load_env()
