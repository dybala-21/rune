"""Tests for rune.identity.resolver — ported from resolver.test.ts."""

from unittest.mock import MagicMock

import pytest

from rune.identity.resolver import IdentityResolver


@pytest.fixture
def resolver(tmp_path):
    """Create a resolver with a temp DB."""
    db_path = tmp_path / "identity.db"
    r = IdentityResolver(db_path=db_path)
    yield r
    r.close()


class TestTuiCliResolution:
    """TUI/CLI channel resolution tests."""

    def test_resolves_tui_to_local_user(self, resolver):
        assert resolver.resolve_identity("tui", "any_sender") == "local_user"

    def test_resolves_cli_to_local_user(self, resolver):
        assert resolver.resolve_identity("cli", "any_sender") == "local_user"

    def test_scopes_local_user_by_workspace_path(self, resolver):
        result = resolver.resolve_identity("tui", "/Users/example/workspace/rune")
        assert result == "local_user:/Users/example/workspace/rune"

    def test_does_not_scope_when_sender_is_not_path(self, resolver):
        result = resolver.resolve_identity("tui", "plain-sender")
        assert result == "local_user"


class TestDefaultResolution:
    """Default single-user resolution tests."""

    def test_returns_default_for_unknown_telegram_user(self, resolver):
        assert resolver.resolve_identity("telegram", "12345") == "default"

    def test_returns_default_for_unknown_discord_user(self, resolver):
        assert resolver.resolve_identity("discord", "user#1234") == "default"

    def test_returns_default_for_any_unknown_channel(self, resolver):
        assert resolver.resolve_identity("slack", "U12345") == "default"


class TestDbBackedResolution:
    """DB-backed resolution tests."""

    def test_returns_linked_user_from_db(self, resolver):
        resolver.link("custom_user_42", "telegram", "99999")
        assert resolver.resolve_identity("telegram", "99999") == "custom_user_42"

    def test_caches_resolved_user(self, resolver):
        resolver.link("cached_user", "telegram", "88888")

        first = resolver.resolve_identity("telegram", "88888")
        assert first == "cached_user"

        # Clear cache to verify DB fallback, then re-cache
        resolver.clear_cache()
        second = resolver.resolve_identity("telegram", "88888")
        assert second == "cached_user"

    def test_handles_db_errors_gracefully(self, resolver):
        # Force a DB error by closing the connection
        resolver._db = MagicMock()
        resolver._db.execute.side_effect = Exception("DB error")
        # Clear cache so it actually hits DB
        resolver._cache.clear()
        result = resolver.resolve_identity("telegram", "77777")
        assert result == "default"


class TestIdentityLinking:
    """Tests for link()."""

    def test_persists_identity_mapping(self, resolver):
        resolver.link("my_user_id", "telegram", "12345")
        assert resolver.resolve_identity("telegram", "12345") == "my_user_id"

    def test_updates_cache_after_linking(self, resolver):
        resolver.link("linked_user", "discord", "abc")
        assert resolver.resolve_identity("discord", "abc") == "linked_user"

    def test_allows_relinking_to_different_user(self, resolver):
        resolver.link("user_a", "telegram", "555")
        assert resolver.resolve_identity("telegram", "555") == "user_a"

        resolver.link("user_b", "telegram", "555")
        assert resolver.resolve_identity("telegram", "555") == "user_b"

    def test_handles_link_errors_gracefully(self, resolver):
        # Force DB to be a broken mock
        resolver._db = MagicMock()
        resolver._db.execute.side_effect = Exception("Write failed")
        # Should not raise
        resolver.link("user", "telegram", "999")


class TestCrossChannel:
    """Cross-channel identity tests."""

    def test_maintains_separate_identities_per_channel(self, resolver):
        tg = resolver.resolve_identity("telegram", "user1")
        dc = resolver.resolve_identity("discord", "user1")
        assert tg == "default"
        assert dc == "default"

    def test_allows_linking_different_channels_to_same_user(self, resolver):
        resolver.link("unified_user", "telegram", "111")
        resolver.link("unified_user", "discord", "aaa")
        assert resolver.resolve_identity("telegram", "111") == "unified_user"
        assert resolver.resolve_identity("discord", "aaa") == "unified_user"
