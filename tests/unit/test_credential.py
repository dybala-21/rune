"""Tests for credential module."""

from __future__ import annotations

from rune.capabilities.credential import (
    KNOWN_CREDENTIALS,
    CredentialSaveParams,
)


def test_known_credentials_mapping():
    """'openai' -> 'OPENAI_API_KEY'."""
    assert KNOWN_CREDENTIALS["openai"] == "OPENAI_API_KEY"
    assert KNOWN_CREDENTIALS["anthropic"] == "ANTHROPIC_API_KEY"
    assert KNOWN_CREDENTIALS["github"] == "GITHUB_TOKEN"
    assert "brave" in KNOWN_CREDENTIALS


def test_credential_save_params():
    """CredentialSaveParams validates fields."""
    params = CredentialSaveParams(key="openai", value="sk-abc123")
    assert params.key == "openai"
    assert params.value == "sk-abc123"
    assert params.scope == "user"

    params2 = CredentialSaveParams(key="github", value="ghp_xxx", scope="project")
    assert params2.scope == "project"
