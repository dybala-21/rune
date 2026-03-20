"""Tests for rune.ui.clipboard — cross-platform clipboard copy."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from rune.ui.clipboard import (
    _detect_clipboard_cmd,
    clipboard_available,
    copy_to_clipboard,
)

# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


class TestClipboardMacOS:
    """Clipboard detection and copy on macOS (darwin)."""

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which", return_value="/usr/bin/pbcopy")
    def test_detect_pbcopy_on_macos(self, mock_which, mock_sys):
        mock_sys.platform = "darwin"
        cmd = _detect_clipboard_cmd()
        assert cmd == ["pbcopy"]

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which", return_value=None)
    def test_unavailable_when_pbcopy_missing(self, mock_which, mock_sys):
        mock_sys.platform = "darwin"
        assert _detect_clipboard_cmd() is None

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which", return_value="/usr/bin/pbcopy")
    def test_clipboard_available_on_macos(self, mock_which, mock_sys):
        mock_sys.platform = "darwin"
        assert clipboard_available() is True

    @patch("rune.ui.clipboard.subprocess.run")
    @patch("rune.ui.clipboard._detect_clipboard_cmd", return_value=["pbcopy"])
    def test_copy_to_clipboard_success(self, mock_detect, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = copy_to_clipboard("hello world")
        assert result is True
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["input"] == b"hello world"

    @patch("rune.ui.clipboard.subprocess.run", side_effect=subprocess.SubprocessError("fail"))
    @patch("rune.ui.clipboard._detect_clipboard_cmd", return_value=["pbcopy"])
    def test_copy_to_clipboard_failure(self, mock_detect, mock_run):
        result = copy_to_clipboard("hello")
        assert result is False


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------


class _FakePlatform(str):
    """A string subclass that supports both == and .startswith() correctly."""
    pass


class TestClipboardLinux:
    """Clipboard detection on Linux."""

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which")
    def test_prefer_xclip_on_linux(self, mock_which, mock_sys):
        mock_sys.platform = _FakePlatform("linux")
        mock_which.side_effect = lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None
        cmd = _detect_clipboard_cmd()
        assert cmd == ["xclip", "-selection", "clipboard"]

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which")
    def test_fallback_to_xsel_on_linux(self, mock_which, mock_sys):
        mock_sys.platform = _FakePlatform("linux")

        def which_side(cmd):
            if cmd == "xsel":
                return "/usr/bin/xsel"
            return None

        mock_which.side_effect = which_side
        cmd = _detect_clipboard_cmd()
        assert cmd == ["xsel", "--clipboard", "--input"]

    @patch("rune.ui.clipboard.sys")
    @patch("rune.ui.clipboard.shutil.which", return_value=None)
    def test_unavailable_when_no_linux_tools(self, mock_which, mock_sys):
        mock_sys.platform = _FakePlatform("linux")
        assert _detect_clipboard_cmd() is None


# ---------------------------------------------------------------------------
# Unsupported platform
# ---------------------------------------------------------------------------


class TestClipboardUnsupported:
    """Clipboard on unsupported platform."""

    @patch("rune.ui.clipboard._detect_clipboard_cmd", return_value=None)
    def test_clipboard_not_available(self, mock_detect):
        assert clipboard_available() is False

    @patch("rune.ui.clipboard._detect_clipboard_cmd", return_value=None)
    def test_copy_returns_false(self, mock_detect):
        assert copy_to_clipboard("text") is False
