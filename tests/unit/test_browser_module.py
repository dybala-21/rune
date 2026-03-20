"""Tests for browser module."""

from __future__ import annotations

from rune.browser.auto_detect import detect_browser_profile


def test_detect_managed():
    """URL in goal -> managed."""
    result = detect_browser_profile("Go to https://example.com and take a screenshot")
    assert result == "managed"


def test_detect_managed_keyword():
    """'browse' -> managed."""
    result = detect_browser_profile("browse the documentation page")
    assert result == "managed"


def test_detect_default():
    """No keywords -> managed (default)."""
    result = detect_browser_profile("do something unrelated")
    assert result == "managed"
