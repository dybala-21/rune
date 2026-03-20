"""Tests for cron capability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rune.capabilities.cron import CronCreateParams, CronDeleteParams, CronListParams


def test_cron_create_params():
    """CronCreateParams validates required fields."""
    params = CronCreateParams(
        name="backup",
        schedule="0 3 * * *",
        command="tar czf backup.tar.gz /data",
        description="Daily backup",
    )
    assert params.name == "backup"
    assert params.schedule == "0 3 * * *"
    assert params.command == "tar czf backup.tar.gz /data"
    assert params.description == "Daily backup"

    # Missing required field
    with pytest.raises(ValidationError):
        CronCreateParams(name="test")  # type: ignore[call-arg]


def test_cron_list_params():
    """CronListParams defaults."""
    params = CronListParams()
    assert params.status == ""

    params2 = CronListParams(status="active")
    assert params2.status == "active"


def test_cron_delete_params():
    """CronDeleteParams requires name."""
    params = CronDeleteParams(name="backup")
    assert params.name == "backup"

    with pytest.raises(ValidationError):
        CronDeleteParams()  # type: ignore[call-arg]
