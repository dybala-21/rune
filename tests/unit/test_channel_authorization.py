"""Every inbound channel must gate on allowed_users.

Telegram already refused unauthorized senders in its handler; Discord and
Slack accepted an allowed_users list nowhere and enforced it never, so an
open bot token let anyone in the server or workspace drive the agent. These
tests pin the gate on the base-class check all three now share.
"""
from __future__ import annotations

from rune.channels.discord import DiscordAdapter
from rune.channels.slack import SlackAdapter


class TestConstructorAcceptsAllowedUsers:
    def test_discord(self):
        a = DiscordAdapter(token="x", allowed_users=["42"])
        assert a.check_authorization("42") is True
        assert a.check_authorization("99") is False

    def test_slack(self):
        a = SlackAdapter(bot_token="x", app_token="y", allowed_users=["U1"])
        assert a.check_authorization("U1") is True
        assert a.check_authorization("U2") is False


class TestOpenByDefaultOnlyWithoutList:
    # No list = open access is the documented base behavior; the fix is that
    # a configured list is now actually consulted.
    def test_discord_open_when_unset(self):
        assert DiscordAdapter(token="x").check_authorization("anyone") is True

    def test_slack_open_when_unset(self):
        assert SlackAdapter(bot_token="x", app_token="y") \
            .check_authorization("anyone") is True
