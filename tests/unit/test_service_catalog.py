"""Tests for rune.services.catalog — ported from catalog.test.ts."""

import pytest

from rune.services.catalog import (
    BUILTIN_SERVICES,
    DEFAULT_WRITE_PATTERN,
    AuthConfig,
    AuthType,
    MCPServerConfig,
    ServiceCategory,
    ServiceDefinition,
    detect_service_mention,
    find_service,
    find_service_by_id,
    get_service_catalog,
    register_custom_service,
    reset_custom_services,
)


@pytest.fixture(autouse=True)
def _reset_custom():
    yield
    reset_custom_services()


class TestFindService:
    """Tests for find_service()."""

    def test_finds_google_calendar_by_id(self):
        result = find_service("google-calendar")
        assert result is not None
        assert result.id == "google-calendar"

    def test_finds_google_calendar_by_tag(self):
        result = find_service("calendar")
        assert result is not None
        assert result.id == "google-calendar"

    def test_finds_notion_by_name(self):
        result = find_service("notion")
        assert result is not None
        assert result.id == "notion"

    def test_finds_todoist_by_name(self):
        result = find_service("todoist")
        assert result is not None
        assert result.id == "todoist"

    def test_finds_github_by_tag(self):
        result = find_service("git")
        assert result is not None
        assert result.id == "github"

    def test_returns_none_for_unknown_service(self):
        assert find_service("nonexistent_service_xyz") is None

    def test_is_case_insensitive(self):
        result = find_service("NOTION")
        assert result is not None
        assert result.id == "notion"


class TestFindServiceById:
    """Tests for find_service_by_id()."""

    def test_finds_by_exact_id(self):
        result = find_service_by_id("google-calendar")
        assert result is not None
        assert result.name == "Google Calendar"

    def test_returns_none_for_nonexistent_id(self):
        assert find_service_by_id("nonexistent") is None


class TestBuiltinServices:
    """Tests for BUILTIN_SERVICES."""

    def test_has_builtin_services(self):
        assert len(BUILTIN_SERVICES) > 0

    def test_each_service_has_valid_auth_spec(self):
        for svc in BUILTIN_SERVICES:
            assert svc.auth.type in ("none", "api-key", "oauth")
            assert len(svc.auth.setup_steps) > 0
            if svc.auth.type != "none":
                assert len(svc.auth.credentials) > 0


class TestRegisterCustomService:
    """Tests for register_custom_service()."""

    def test_adds_custom_service_to_catalog(self):
        custom = ServiceDefinition(
            id="custom-test",
            name="Custom Test",
            description="Test service",
            category=ServiceCategory.OTHER,
            mcp_server=MCPServerConfig(
                package="@test/mcp-test",
                command="npx",
                args=["-y", "@test/mcp-test"],
            ),
            auth=AuthConfig(
                type=AuthType.NONE,
                setup_steps=["No setup needed"],
            ),
            tags=["test"],
            aliases=["custom"],
        )
        register_custom_service(custom)
        catalog = get_service_catalog()
        assert len(catalog) == len(BUILTIN_SERVICES) + 1
        assert find_service("custom-test") is not None


class TestDetectServiceMention:
    """Tests for detect_service_mention()."""

    def test_detects_by_english_alias(self):
        result = detect_service_mention("check my calendar")
        assert result is not None
        assert result.id == "google-calendar"

    def test_detects_by_service_id(self):
        result = detect_service_mention("connect google-calendar")
        assert result is not None
        assert result.id == "google-calendar"

    def test_returns_none_for_unrelated_text(self):
        assert detect_service_mention("hello world refactor code") is None

    def test_detects_weather_service(self):
        result = detect_service_mention("weather forecast today")
        assert result is not None
        assert result.id == "openweathermap"

    def test_is_case_insensitive(self):
        result = detect_service_mention("NOTION page check")
        assert result is not None
        assert result.id == "notion"

    def test_detects_custom_services(self):
        register_custom_service(ServiceDefinition(
            id="my-chat",
            name="MyChat",
            description="Chat service",
            category=ServiceCategory.COMMUNICATION,
            mcp_server=MCPServerConfig(package="@t/chat", command="npx", args=["-y", "@t/chat"]),
            auth=AuthConfig(type=AuthType.API_KEY, credentials=[], setup_steps=["Setup"]),
            tags=["chat"],
            aliases=["mychat"],
        ))
        result = detect_service_mention("send mychat message")
        assert result is not None
        assert result.id == "my-chat"


class TestDefaultWritePattern:
    """Tests for DEFAULT_WRITE_PATTERN."""

    def test_matches_create(self):
        assert DEFAULT_WRITE_PATTERN.search("create-event") is not None

    def test_matches_delete(self):
        assert DEFAULT_WRITE_PATTERN.search("delete-item") is not None

    def test_matches_send(self):
        assert DEFAULT_WRITE_PATTERN.search("send-message") is not None

    def test_does_not_match_list(self):
        assert DEFAULT_WRITE_PATTERN.search("list-events") is None

    def test_does_not_match_get(self):
        assert DEFAULT_WRITE_PATTERN.search("get-freebusy") is None
