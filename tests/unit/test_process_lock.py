"""Tests for rune.daemon.process_lock — PID file-based process locking."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from rune.daemon.process_lock import (
    acquire_lock,
    get_lock_owner,
    is_locked,
    is_process_running,
    release_lock,
)


class TestIsProcessRunning:
    def test_current_process_is_running(self):
        assert is_process_running(os.getpid()) is True

    def test_invalid_pid_returns_false(self):
        assert is_process_running(0) is False
        assert is_process_running(-1) is False


class TestAcquireLock:
    def test_acquires_new_lock(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        result = acquire_lock(lock_path, pid=4242)
        try:
            assert result == lock_path
            content = lock_path.read_text().strip()
            assert content == "4242"
        finally:
            release_lock(lock_path)

    def test_replaces_stale_lock(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        lock_path.write_text("99999\n")  # non-existent PID

        with patch("rune.daemon.process_lock.is_process_running", return_value=False):
            acquire_lock(lock_path, pid=2222)
            try:
                content = lock_path.read_text().strip()
                assert content == "2222"
            finally:
                release_lock(lock_path)

    def test_raises_when_owner_still_running(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        lock_path.write_text("3333\n")

        with patch("rune.daemon.process_lock.is_process_running", return_value=True):
            with pytest.raises(RuntimeError, match="Process lock already held by pid 3333"):
                acquire_lock(lock_path, pid=4444)


class TestReleaseLock:
    def test_release_removes_lock_file(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        acquire_lock(lock_path, pid=os.getpid())
        release_lock(lock_path)
        assert not lock_path.exists()

    def test_release_ignores_missing_lock(self, tmp_path):
        lock_path = tmp_path / "missing.lock"
        # Should not raise
        release_lock(lock_path)


class TestIsLocked:
    def test_returns_false_when_no_file(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        assert is_locked(lock_path) is False

    def test_returns_true_when_held(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        acquire_lock(lock_path, pid=os.getpid())
        try:
            assert is_locked(lock_path) is True
        finally:
            release_lock(lock_path)


class TestGetLockOwner:
    def test_returns_none_when_no_file(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        assert get_lock_owner(lock_path) is None

    def test_returns_pid_when_locked(self, tmp_path):
        lock_path = tmp_path / "daemon.lock"
        acquire_lock(lock_path, pid=os.getpid())
        try:
            owner = get_lock_owner(lock_path)
            assert owner == os.getpid()
        finally:
            release_lock(lock_path)
