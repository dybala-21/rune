"""Service catalog for RUNE.

Ported from src/services/catalog.ts - registry of built-in and custom
external service definitions with metadata, auth config, and aliases
for service-mention detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ============================================================================
# Types
# ============================================================================

class ServiceCategory(StrEnum):
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    DATA = "data"
    MEDIA = "media"
    LIFESTYLE = "lifestyle"
    SYSTEM = "system"
    OTHER = "other"


class AuthType(StrEnum):
    NONE = "none"
    API_KEY = "api-key"
    OAUTH = "oauth"


class CredentialType(StrEnum):
    FILE_PATH = "file-path"
    API_KEY = "api-key"
    TOKEN = "token"
    URL = "url"
    STRING = "string"


@dataclass(slots=True)
class ServiceCredentialSpec:
    """Describes a single credential required by a service."""

    key: str
    """Environment variable name (e.g., 'GOOGLE_OAUTH_CREDENTIALS')."""

    label: str
    """User-facing display name."""

    type: CredentialType

    help_url: str | None = None
    """URL to credential-provisioning instructions."""

    help_text: str | None = None
    """Short guidance (1-2 lines)."""


class OAuthTriggerType(StrEnum):
    TOOL = "tool"
    CLI_AUTH = "cli-auth"
    STDERR = "stderr"


@dataclass(slots=True)
class OAuthTriggerConfig:
    """How the OAuth browser flow is initiated."""

    type: OAuthTriggerType

    # -- type == 'tool' --
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    auth_url_field: str = "auth_url"
    already_auth_patterns: list[str] = field(default_factory=list)

    # -- type == 'cli-auth' --
    auth_command: str | None = None
    auth_args: list[str] | None = None


@dataclass(slots=True)
class OAuthAutoConfig:
    """OAuth automation settings for services with ``auth.type == 'oauth'``."""

    auto_flow: bool = False
    credentials_filename: str = ""
    token_path_env_key: str | None = None
    trigger: OAuthTriggerConfig | None = None
    oauth_url_pattern: re.Pattern[str] | None = None
    token_saved_pattern: re.Pattern[str] | None = None
    token_default_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MCPServerConfig:
    """MCP server connection details."""

    package: str
    command: str
    args: list[str]
    enable_tools: list[str] | None = None


@dataclass(slots=True)
class AuthConfig:
    """Authentication requirements for a service."""

    type: AuthType
    credentials: list[ServiceCredentialSpec] = field(default_factory=list)
    setup_steps: list[str] = field(default_factory=list)
    oauth: OAuthAutoConfig | None = None


@dataclass(slots=True)
class ServiceDefinition:
    """Complete definition of an external service."""

    id: str
    name: str
    description: str
    category: ServiceCategory
    mcp_server: MCPServerConfig
    auth: AuthConfig
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    write_patterns: re.Pattern[str] | None = None
    skill_template: str | None = None


# ============================================================================
# Constants
# ============================================================================

#: Default write-operation detection pattern (shared across all services).
DEFAULT_WRITE_PATTERN: re.Pattern[str] = re.compile(
    r"create|update|delete|remove|send|post|put|patch|write|insert"
    r"|modify|edit|add|move|archive",
    re.IGNORECASE,
)


# ============================================================================
# Built-in services
# ============================================================================

BUILTIN_SERVICES: list[ServiceDefinition] = [
    # -- Productivity --
    ServiceDefinition(
        id="google-calendar",
        name="Google Calendar",
        description="View, create, update, and delete calendar events",
        category=ServiceCategory.PRODUCTIVITY,
        mcp_server=MCPServerConfig(
            package="@cocal/google-calendar-mcp",
            command="npx",
            args=["-y", "@cocal/google-calendar-mcp"],
            enable_tools=[
                "list-calendars", "list-events", "search-events",
                "create-event", "update-event", "delete-event",
                "get-freebusy", "get-current-time",
            ],
        ),
        auth=AuthConfig(
            type=AuthType.OAUTH,
            credentials=[
                ServiceCredentialSpec(
                    key="GOOGLE_OAUTH_CREDENTIALS",
                    label="Google OAuth key file",
                    type=CredentialType.FILE_PATH,
                    help_url="https://console.cloud.google.com/apis/credentials",
                ),
            ],
            setup_steps=[
                "Go to Google Cloud Console (console.cloud.google.com)",
                "Create a project and enable the Calendar API",
                "Create an OAuth 2.0 Desktop client ID",
                "Download the JSON key file",
                "Provide the path to the downloaded file",
            ],
            oauth=OAuthAutoConfig(
                auto_flow=True,
                credentials_filename="google-calendar-oauth.json",
                token_path_env_key="GOOGLE_CALENDAR_MCP_TOKEN_PATH",
                trigger=OAuthTriggerConfig(
                    type=OAuthTriggerType.TOOL,
                    tool_name="manage-accounts",
                    tool_arguments={"action": "add", "account_id": "default"},
                    auth_url_field="auth_url",
                    already_auth_patterns=["already_authenticated", "already connected"],
                ),
                oauth_url_pattern=re.compile(
                    r"https://accounts\.google\.com/o/oauth2[^\s\"'\]>)]+"
                ),
                token_saved_pattern=re.compile(r"Tokens saved successfully", re.IGNORECASE),
                token_default_paths=["~/.config/google-calendar-mcp/tokens.json"],
            ),
        ),
        tags=["calendar", "schedule", "meetings"],
        aliases=["calendar", "schedule", "meeting", "meetings"],
        skill_template="schedule-manager",
    ),
    ServiceDefinition(
        id="notion",
        name="Notion",
        description="Query and manage pages and databases",
        category=ServiceCategory.PRODUCTIVITY,
        mcp_server=MCPServerConfig(
            package="@notionhq/notion-mcp-server",
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="OPENAPI_MCP_HEADERS",
                    label="Notion API key",
                    type=CredentialType.API_KEY,
                    help_url="https://www.notion.so/my-integrations",
                ),
            ],
            setup_steps=[
                "Open Notion Settings -> My Integrations",
                "Click 'New integration' and configure permissions",
                "Copy the API key",
            ],
        ),
        tags=["notes", "wiki", "database", "project"],
        aliases=["notion", "notes", "wiki"],
    ),
    ServiceDefinition(
        id="todoist",
        name="Todoist",
        description="Task management, projects, and to-do tracking",
        category=ServiceCategory.PRODUCTIVITY,
        mcp_server=MCPServerConfig(
            package="@greirson/mcp-todoist",
            command="npx",
            args=["-y", "@greirson/mcp-todoist"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="TODOIST_API_TOKEN",
                    label="Todoist API token",
                    type=CredentialType.API_KEY,
                    help_url="https://todoist.com/app/settings/integrations/developer",
                ),
            ],
            setup_steps=[
                "Open Todoist Settings -> Integrations -> Developer",
                "Copy the API token",
            ],
        ),
        tags=["todo", "tasks", "project"],
        aliases=["todoist", "todo", "task", "tasks"],
    ),
    ServiceDefinition(
        id="deepl",
        name="DeepL",
        description="High-quality translation (500k chars/month free)",
        category=ServiceCategory.PRODUCTIVITY,
        mcp_server=MCPServerConfig(
            package="deepl-mcp-server",
            command="npx",
            args=["-y", "deepl-mcp-server"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="DEEPL_API_KEY",
                    label="DeepL API key",
                    type=CredentialType.API_KEY,
                    help_url="https://www.deepl.com/pro-api",
                ),
            ],
            setup_steps=[
                "Create a free API account at deepl.com/pro-api",
                "Copy the API key from account settings",
            ],
        ),
        tags=["translate", "translation", "language"],
        aliases=["deepl", "translate", "translation"],
    ),

    # -- Development --
    ServiceDefinition(
        id="github",
        name="GitHub",
        description="Repository, issue, and pull request management",
        category=ServiceCategory.DEVELOPMENT,
        mcp_server=MCPServerConfig(
            package="@modelcontextprotocol/server-github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="GITHUB_PERSONAL_ACCESS_TOKEN",
                    label="GitHub Personal Access Token",
                    type=CredentialType.API_KEY,
                    help_url="https://github.com/settings/tokens",
                ),
            ],
            setup_steps=[
                "Go to GitHub Settings -> Developer Settings -> Personal Access Tokens",
                "Generate a new token with required scopes (repo, read:org, etc.)",
                "Copy the token",
            ],
        ),
        tags=["git", "repo", "issues", "pr"],
        aliases=["github", "gh", "issue", "pull request", "repo"],
    ),

    # -- Communication --
    ServiceDefinition(
        id="slack",
        name="Slack",
        description="Channel, DM, and message management",
        category=ServiceCategory.COMMUNICATION,
        mcp_server=MCPServerConfig(
            package="@modelcontextprotocol/server-slack",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-slack"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="SLACK_BOT_TOKEN",
                    label="Slack Bot Token",
                    type=CredentialType.API_KEY,
                    help_url="https://api.slack.com/apps",
                ),
                ServiceCredentialSpec(
                    key="SLACK_TEAM_ID",
                    label="Slack Team ID",
                    type=CredentialType.STRING,
                ),
            ],
            setup_steps=[
                "Create a Slack App at api.slack.com/apps",
                "Add Bot Token Scopes (channels:history, chat:write, etc.)",
                "Install the app to your workspace",
                "Copy the Bot Token (xoxb-...) and Team ID",
            ],
        ),
        tags=["chat", "messaging", "team"],
        aliases=["slack"],
    ),

    # -- Lifestyle --
    ServiceDefinition(
        id="google-maps",
        name="Google Maps",
        description="Directions, place search, distance calculation",
        category=ServiceCategory.LIFESTYLE,
        mcp_server=MCPServerConfig(
            package="@modelcontextprotocol/server-google-maps",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-google-maps"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="GOOGLE_MAPS_API_KEY",
                    label="Google Maps API key",
                    type=CredentialType.API_KEY,
                    help_url="https://console.cloud.google.com/apis/credentials",
                ),
            ],
            setup_steps=[
                "Go to Google Cloud Console",
                "Enable Maps JavaScript API and Geocoding API",
                "Create an API key",
            ],
        ),
        tags=["maps", "directions", "places", "geocoding"],
        aliases=["maps", "google maps", "directions", "navigation"],
    ),
    ServiceDefinition(
        id="openweathermap",
        name="Weather",
        description="Current weather, forecasts, and air quality",
        category=ServiceCategory.LIFESTYLE,
        mcp_server=MCPServerConfig(
            package="mcp-openweathermap",
            command="npx",
            args=["-y", "mcp-openweathermap"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="OPENWEATHER_API_KEY",
                    label="OpenWeatherMap API key",
                    type=CredentialType.API_KEY,
                    help_url="https://openweathermap.org/api",
                ),
            ],
            setup_steps=[
                "Create a free account at openweathermap.org",
                "Copy the API key from the API keys tab",
            ],
        ),
        tags=["weather", "forecast", "temperature"],
        aliases=["weather", "forecast", "temperature"],
    ),

    # -- Media --
    ServiceDefinition(
        id="spotify",
        name="Spotify",
        description="Music playback, search, and playlist management",
        category=ServiceCategory.MEDIA,
        mcp_server=MCPServerConfig(
            package="mcp-spotify",
            command="npx",
            args=["-y", "mcp-spotify"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="SPOTIFY_CLIENT_ID",
                    label="Spotify Client ID",
                    type=CredentialType.API_KEY,
                    help_url="https://developer.spotify.com/dashboard",
                ),
                ServiceCredentialSpec(
                    key="SPOTIFY_CLIENT_SECRET",
                    label="Spotify Client Secret",
                    type=CredentialType.API_KEY,
                ),
            ],
            setup_steps=[
                "Go to developer.spotify.com/dashboard",
                "Create an app and add redirect URI http://localhost:8888/callback",
                "Copy Client ID and Client Secret",
            ],
        ),
        tags=["music", "audio", "playlist"],
        aliases=["spotify", "music", "playlist"],
    ),

    # -- Data / Search --
    ServiceDefinition(
        id="perplexity",
        name="Perplexity",
        description="AI search engine with sourced answers",
        category=ServiceCategory.DATA,
        mcp_server=MCPServerConfig(
            package="perplexity-mcp",
            command="npx",
            args=["-y", "perplexity-mcp"],
        ),
        auth=AuthConfig(
            type=AuthType.API_KEY,
            credentials=[
                ServiceCredentialSpec(
                    key="PERPLEXITY_API_KEY",
                    label="Perplexity API key",
                    type=CredentialType.API_KEY,
                    help_url="https://docs.perplexity.ai",
                ),
            ],
            setup_steps=[
                "Create an account at perplexity.ai",
                "Generate an API key in Settings -> API",
            ],
        ),
        tags=["search", "research", "answers"],
        aliases=["perplexity", "search", "research"],
    ),

    # -- System --
    ServiceDefinition(
        id="apple-shortcuts",
        name="Apple Shortcuts",
        description="Run macOS/iOS shortcuts",
        category=ServiceCategory.SYSTEM,
        mcp_server=MCPServerConfig(
            package="shortcuts-mcp",
            command="npx",
            args=["-y", "shortcuts-mcp"],
        ),
        auth=AuthConfig(type=AuthType.NONE, setup_steps=[
            "Available automatically on macOS",
            "Grant permissions in System Settings -> Privacy -> Automation",
        ]),
        tags=["shortcuts", "automation", "macos"],
        aliases=["shortcuts", "automation", "automator"],
    ),
]


# ============================================================================
# Runtime registry
# ============================================================================

_custom_services: list[ServiceDefinition] = []


def get_service_catalog() -> list[ServiceDefinition]:
    """Return the full service catalog (built-in + custom)."""
    return [*BUILTIN_SERVICES, *_custom_services]


def find_service(query: str) -> ServiceDefinition | None:
    """Find a service by fuzzy query (id, name, tag, or alias)."""
    q = query.lower()
    for svc in get_service_catalog():
        if svc.id == q:
            return svc
        if q in svc.name.lower():
            return svc
        if any(q in t for t in svc.tags):
            return svc
        if any(a.lower() == q or q in a.lower() for a in svc.aliases):
            return svc
    return None


def find_service_by_id(service_id: str) -> ServiceDefinition | None:
    """Find a service by exact ID."""
    for svc in get_service_catalog():
        if svc.id == service_id:
            return svc
    return None


def register_custom_service(service: ServiceDefinition) -> None:
    """Add a custom service definition to the catalog."""
    _custom_services.append(service)


def reset_custom_services() -> None:
    """Clear custom services (for testing)."""
    _custom_services.clear()


# ============================================================================
# Catalog-driven service detection
# ============================================================================

def detect_service_mention(goal: str) -> ServiceDefinition | None:
    """Detect whether *goal* mentions a cataloged service.

    Checks aliases first (supports multi-word aliases), then falls
    back to ID matching.

    Returns the matched ``ServiceDefinition`` or ``None``.
    """
    lower = goal.lower()
    for svc in get_service_catalog():
        for alias in svc.aliases:
            if alias.lower() in lower:
                return svc
        if svc.id in lower:
            return svc
    return None
