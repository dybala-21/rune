"""A hosted machine reads schedules in its owner's zone, and still knows who to notify after a restart.

Hosted machines used to run in UTC, so "every day at 7" from a chat in Seoul fired at 4 pm there,
and the Telegram chat that results go to was forgotten whenever the machine restarted.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from rune.capabilities import cron
from rune.channels.registry import ChannelRegistry
from rune.channels.types import IncomingMessage
from rune.daemon import external_schedule as ext
from rune.daemon.gateway import ChannelGateway, GatewayNotification
from tests.unit.test_external_schedule import _drain, job  # noqa: F401 (fixture)


@pytest.fixture
def zone(monkeypatch):
    """Run in a zone the way a hosted machine's TZ does, and put the process back after."""
    before = os.environ.get("TZ")

    def use(name: str | None) -> None:
        if name is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", name)
        time.tzset()

    yield use
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


@pytest.mark.parametrize(("tz", "reported"), [("Asia/Seoul", "Asia/Seoul"), (None, None), ("Not/AZone", None),
                                              (":/etc/localtime", None)])
async def test_cron_list_reports_the_zone_schedules_are_read_in(job, monkeypatch, zone, tz, reported):  # noqa: F811
    from rune.api.handlers import cron as api

    _job_id, store = job
    monkeypatch.setattr(api, "get_memory_store", lambda: store)
    zone(tz)
    assert (await api.list_cron_jobs()).timezone == reported


async def test_nine_in_the_morning_means_nine_where_the_owner_is(job, monkeypatch, zone):  # noqa: F811
    job_id, _store = job  # "0 9 * * *"
    zone("Asia/Seoul")
    seoul_nine = int(datetime(2026, 11, 16, 0, 0, tzinfo=UTC).timestamp() // 60)  # 09:00 KST
    utc_nine = seoul_nine + 9 * 60
    monkeypatch.setattr(cron, "execute_cron_job", AsyncMock(return_value=None))

    monkeypatch.setattr("time.time", lambda: utc_nine * 60 + 5)
    with pytest.raises(ValueError, match="not on this job's schedule"):
        await ext.fire(job_id, utc_nine)

    monkeypatch.setattr("time.time", lambda: seoul_nine * 60 + 5)
    started = await ext.fire(job_id, seoul_nine)
    await _drain()
    assert started["occurrenceId"] == ext.occurrence_id(job_id, seoul_nine)
    cron.execute_cron_job.assert_awaited_once()


async def test_creating_a_job_says_which_zone_the_time_is_in(job, zone):  # noqa: F811
    zone("Asia/Seoul")
    created = await cron.cron_create(cron.CronCreateParams(name="brief", schedule="0 7 * * *", goal="Brief me"))
    assert created.success and "Schedule: 0 7 * * * (Asia/Seoul)" in created.output


class _Adapter:
    name = "telegram"
    status = "connected"

    def __init__(self) -> None:
        self.on_message = None
        self.sent: list[tuple[str, str]] = []

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, channel_id, message) -> None: ...

    async def send_notification(self, recipient, text) -> None:
        self.sent.append((recipient, text))


class _Scheduler:
    async def execute(self, goal, sender_id):
        return "ok"


async def test_the_chat_results_go_to_survives_a_restart(tmp_path):
    path = tmp_path / "data" / "channel-recipients.json"
    registry = ChannelRegistry()
    adapter = _Adapter()
    registry.register(adapter)
    first = ChannelGateway(registry, agent_scheduler=_Scheduler(), recipients_path=path)
    await first.start()
    try:
        await adapter.on_message(IncomingMessage(channel_id="777", sender_id="777", text="hi", attachments=[],
                                                 metadata={"channel_name": "telegram"}))
        await asyncio.sleep(0.05)
    finally:
        await first.stop()
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    restarted = ChannelGateway(registry, recipients_path=path)
    await restarted.route_notification(GatewayNotification(title="Morning", body="brief", channel="telegram"))
    assert adapter.sent and adapter.sent[0][0] == "777"


def test_an_unreadable_recipient_file_is_ignored(tmp_path):
    path = tmp_path / "channel-recipients.json"
    path.write_text("{not json")
    assert ChannelGateway(ChannelRegistry(), recipients_path=path)._default_recipients == {}
    path.write_text('{"telegram": 5, "slack": "U1"}')
    gateway = ChannelGateway(ChannelRegistry(), recipients_path=path, default_recipients={"slack": "U2"})
    assert gateway._default_recipients == {"slack": "U2"}, "given recipients win; malformed entries are dropped"
