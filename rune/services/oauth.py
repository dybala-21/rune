"""OAuth2 flow support for RUNE services.

Ported from src/services/oauth.ts - handles OAuth authorization URL
generation, token exchange, refresh, browser-based flows, and secure
token storage for MCP-backed services.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rune.services.catalog import OAuthTriggerType, ServiceDefinition
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

CREDENTIALS_DIR: Path = rune_home() / "credentials"
"""Directory for stored OAuth credentials and tokens."""

_OAUTH_CONNECT_TIMEOUT: float = 180.0  # 3 minutes

#: Generic OAuth URL fallback pattern (when service has no specific pattern).
_GENERIC_OAUTH_URL_PATTERN = (
    r"https?://[^\s\"'\]>)]+(?:oauth|authorize|login|auth)[^\s\"'\]>)]*"
)


# ============================================================================
# Types
# ============================================================================

@dataclass(slots=True)
class OAuthSetupResult:
    """Result of an OAuth setup attempt."""

    success: bool
    message: str = ""
    tool_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class OAuthSetupOptions:
    """Options for :func:`perform_oauth_setup`."""

    channel: str | None = None
    """Channel from which the request originated (tui, cli, telegram, etc.)."""


# ============================================================================
# Helpers
# ============================================================================

def _is_local_channel(channel: str | None) -> bool:
    """Return ``True`` if the channel can open a local browser."""
    return not channel or channel in ("cli", "tui", "local")


def _expand_home(p: str) -> str:
    return p.replace("~", str(Path.home()), 1) if p.startswith("~") else p


def _open_browser(url: str) -> None:
    """Open *url* in the system's default browser."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url])  # noqa: S603
        elif system == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "", url])  # noqa: S603
        else:
            subprocess.Popen(["xdg-open", url])  # noqa: S603
        log.debug("browser_opened", url=url[:80])
    except OSError as exc:
        log.warning("browser_open_failed", error=str(exc))


# ============================================================================
# Credential management
# ============================================================================

def prepare_credentials(service: ServiceDefinition) -> Path:
    """Locate or prepare OAuth credentials for *service*.

    Resolution order:
    1. ``~/.rune/credentials/{filename}`` already exists
    2. Environment variable points to a valid JSON file -> copy
    3. Raise with guidance

    Returns the absolute path to the credentials file.
    """
    oauth = service.auth.oauth
    if oauth is None:
        raise ValueError(f'Service "{service.id}" has no OAuth config')

    cred_path = CREDENTIALS_DIR / oauth.credentials_filename
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Already present
    if cred_path.exists():
        log.debug("oauth_credentials_found", path=str(cred_path))
        return cred_path

    # 2. Env var -> copy
    cred_key = service.auth.credentials[0].key if service.auth.credentials else None
    if cred_key:
        import os

        env_path = os.environ.get(cred_key)
        if env_path:
            src = Path(env_path)
            try:
                content = src.read_text(encoding="utf-8")
                json_decode(content)  # validate JSON
                cred_path.write_text(content, encoding="utf-8")
                cred_path.chmod(0o600)
                log.debug(
                    "oauth_credentials_copied",
                    src=str(src),
                    dest=str(cred_path),
                )
                return cred_path
            except (OSError, json.JSONDecodeError, ValueError):
                log.warning("oauth_credentials_env_invalid", key=cred_key, path=env_path)

    # 3. Not found
    help_url = (
        service.auth.credentials[0].help_url
        if service.auth.credentials
        else "https://console.cloud.google.com/apis/credentials"
    )
    raise FileNotFoundError(
        f'OAuth credentials not found for "{service.id}".\n'
        f"Place your OAuth JSON file at: {cred_path}\n"
        f"Or set env var: {cred_key or 'N/A'}=/path/to/oauth.json\n"
        f"Get credentials: {help_url}"
    )


def has_existing_tokens(service: ServiceDefinition) -> bool:
    """Return ``True`` if OAuth tokens already exist on disk."""
    token_path = CREDENTIALS_DIR / service.id / "tokens.json"
    return token_path.exists()


def build_oauth_env(
    service: ServiceDefinition, cred_path: Path
) -> dict[str, str]:
    """Build environment variables needed for an OAuth MCP server."""
    oauth = service.auth.oauth
    if oauth is None:
        return {}

    env: dict[str, str] = {}

    # Credential file path
    cred_key = service.auth.credentials[0].key if service.auth.credentials else None
    if cred_key:
        env[cred_key] = str(cred_path)

    # Token storage path (inside RUNE directory)
    if oauth.token_path_env_key:
        token_dir = CREDENTIALS_DIR / service.id
        token_dir.mkdir(parents=True, exist_ok=True)
        env[oauth.token_path_env_key] = str(token_dir / "tokens.json")

    return env


def clear_expired_tokens(service: ServiceDefinition) -> None:
    """Delete all known token files for *service* (before re-auth)."""
    oauth = service.auth.oauth
    paths = [
        CREDENTIALS_DIR / service.id / "tokens.json",
    ]
    if oauth:
        paths.extend(Path(_expand_home(p)) for p in oauth.token_default_paths)

    for token_path in paths:
        try:
            token_path.unlink()
            log.debug("token_cleared", path=str(token_path))
        except FileNotFoundError:
            pass


# ============================================================================
# OAuth setup (main entry point)
# ============================================================================

async def perform_oauth_setup(
    service: ServiceDefinition,
    options: OAuthSetupOptions | None = None,
) -> OAuthSetupResult:
    """Run the full OAuth auto-flow for *service*.

    Dispatches to the appropriate trigger strategy based on the
    service catalog configuration:

    * ``tool`` -- connect MCP server, call a tool to get auth URL
    * ``cli-auth`` -- run a CLI command that handles browser auth
    * ``stderr`` -- connect MCP server, detect URL from stderr

    Parameters
    ----------
    service:
        The ``ServiceDefinition`` (must have ``auth.type == "oauth"``).
    options:
        Optional settings (channel, etc.).

    Returns
    -------
    OAuthSetupResult
        Whether the setup succeeded and the number of available tools.
    """
    opts = options or OAuthSetupOptions()
    oauth = service.auth.oauth

    if oauth is None or not oauth.auto_flow:
        return OAuthSetupResult(
            success=False,
            error="Not an autoFlow OAuth service",
        )

    trigger_type = oauth.trigger.type if oauth.trigger else OAuthTriggerType.STDERR
    is_local = _is_local_channel(opts.channel)

    # Remote channel without tokens -> guide user to local setup
    if not is_local and not has_existing_tokens(service):
        return _build_remote_guide(service, "requires-local-oauth")

    try:
        # 1. Clear expired tokens (local only)
        if is_local:
            clear_expired_tokens(service)

        # 2. Prepare credentials
        cred_path = prepare_credentials(service)

        # 3. Build env
        env = build_oauth_env(service, cred_path)

        # 4. Write MCP config
        from rune.services.connector import ServiceConnector

        success, config_path = ServiceConnector.connect(service, env)
        if not success:
            return OAuthSetupResult(
                success=False,
                error="Failed to configure MCP server",
            )

        # 5. Trigger-specific flow
        if trigger_type == OAuthTriggerType.CLI_AUTH:
            if not has_existing_tokens(service) and is_local:
                cli_result = await _trigger_oauth_via_cli(service, env)
                if not cli_result.success:
                    return cli_result

        # 6. For tool/stderr triggers, browser opening would happen
        #    during MCP server startup (via stderr monitoring or tool call).
        #    In this Python port we perform a token-file poll.
        if is_local and not has_existing_tokens(service):
            if trigger_type == OAuthTriggerType.TOOL:
                log.info("oauth_awaiting_tool_trigger", service=service.id)
            elif trigger_type == OAuthTriggerType.STDERR:
                log.info("oauth_awaiting_stderr_url", service=service.id)

            saved = await _wait_for_token_save(service)
            if not saved:
                return OAuthSetupResult(
                    success=False,
                    message=(
                        "Please complete login in the browser.\n"
                        "Connection timed out. Please try again."
                    ),
                    error="OAuth timeout - user did not complete authorization",
                )

        return OAuthSetupResult(
            success=True,
            message=f"{service.name} connected successfully!",
        )

    except Exception as exc:
        msg = str(exc)
        log.error("oauth_setup_failed", service=service.id, error=msg)

        if not is_local:
            return OAuthSetupResult(
                success=False,
                message=(
                    f"{service.name} connection failed.\n"
                    f"Please connect from your computer:\n\n"
                    f"  rune service connect {service.id}"
                ),
                error=f"OAuth setup failed (remote): {msg}",
            )

        return OAuthSetupResult(success=False, error=f"OAuth setup failed: {msg}")


# ============================================================================
# Trigger: CLI auth
# ============================================================================

async def _trigger_oauth_via_cli(
    service: ServiceDefinition,
    env: dict[str, str],
) -> OAuthSetupResult:
    """Run a CLI command to initiate the OAuth browser flow."""
    trigger = service.auth.oauth and service.auth.oauth.trigger
    if not trigger or not trigger.auth_command or not trigger.auth_args:
        return OAuthSetupResult(success=False, error="No CLI auth config")

    import os as _os

    merged_env = {**_os.environ, **env}

    log.info(
        "cli_auth_started",
        command=trigger.auth_command,
        args=trigger.auth_args,
    )

    proc = await asyncio.create_subprocess_exec(
        trigger.auth_command,
        *trigger.auth_args,
        env=merged_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_OAUTH_CONNECT_TIMEOUT
        )
    except TimeoutError:
        proc.kill()
        return OAuthSetupResult(
            success=False,
            message="CLI auth timed out. Please try again.",
            error="CLI auth timeout",
        )

    if proc.returncode == 0:
        log.info("cli_auth_completed", service=service.id)
        return OAuthSetupResult(success=True)

    return OAuthSetupResult(
        success=False,
        error=f"CLI auth exited with code {proc.returncode}",
    )


# ============================================================================
# Token file polling
# ============================================================================

async def _wait_for_token_save(
    service: ServiceDefinition,
    timeout: float | None = None,
) -> bool:
    """Poll for the token file to appear on disk."""
    token_path = CREDENTIALS_DIR / service.id / "tokens.json"
    deadline = time.monotonic() + (timeout or _OAUTH_CONNECT_TIMEOUT)

    while time.monotonic() < deadline:
        if token_path.exists():
            log.info("token_file_detected", service=service.id)
            return True
        await asyncio.sleep(1.0)

    log.warning("token_save_timeout", service=service.id)
    return False


# ============================================================================
# Remote channel guide
# ============================================================================

def _build_remote_guide(service: ServiceDefinition, reason: str) -> OAuthSetupResult:
    message = (
        f"{service.name} requires browser authentication.\n"
        f"Please run the following on your computer:\n\n"
        f"  rune service connect {service.id}\n\n"
        f"After completing login in the browser, "
        f"the service will be available here."
    )
    return OAuthSetupResult(
        success=False,
        message=message,
        error=f"remote-channel-{reason}",
    )
