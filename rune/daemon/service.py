"""Install a foreground daemon under the OS user service manager."""

from __future__ import annotations

import os
import platform
import plistlib
import subprocess
import sys
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home, rune_logs

log = get_logger(__name__)

_SERVICE_NAME = "com.rune.daemon"
_SYSTEMD_UNIT = "rune-daemon.service"


def install_service(host: str = "127.0.0.1", port: int = 18789) -> None:
    """Install a launchd or systemd service for the current user."""
    system = platform.system()
    from rune.daemon.endpoint import Endpoint, save_endpoint
    endpoint = Endpoint(host, port)
    if system not in ("Darwin", "Linux"):
        raise RuntimeError(f"Service installation not supported on {system}")
    if system == "Darwin":
        _install_launchd(host, port)
    elif system == "Linux":
        _install_systemd(host, port)
    save_endpoint(endpoint)


def uninstall_service() -> None:
    """Remove the RUNE daemon system service."""
    system = platform.system()
    if system == "Darwin":
        _uninstall_launchd()
    elif system == "Linux":
        _uninstall_systemd()
    else:
        raise RuntimeError(f"Service uninstallation not supported on {system}")


def get_service_status() -> str:
    """Return the current service status: running, stopped, or not_installed."""
    system = platform.system()
    if system == "Darwin":
        return _launchd_status()
    elif system == "Linux":
        return _systemd_status()
    return "not_installed"


def start_service() -> None:
    command = (["launchctl", "kickstart", f"gui/{os.getuid()}/{_SERVICE_NAME}"]
               if platform.system() == "Darwin" else ["systemctl", "--user", "start", _SYSTEMD_UNIT])
    subprocess.run(command, check=True, capture_output=True)


# macOS (launchd)

def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_SERVICE_NAME}.plist"


def launchd_config(host: str = "127.0.0.1", port: int = 18789) -> dict:
    from rune.daemon.endpoint import Endpoint
    endpoint = Endpoint(host, port)
    return {
        "Label": _SERVICE_NAME,
        "ProgramArguments": [sys.executable, "-m", "rune.cli.main", "daemon", "run",
                             "--host", endpoint.host, "--port", str(endpoint.port)],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ExitTimeOut": 35,
        "WorkingDirectory": str(Path.home()),
        "StandardOutPath": "/dev/null",
        "StandardErrorPath": "/dev/null",
        "EnvironmentVariables": service_environment(),
    }


def service_environment() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
           "RUNE_HOME": str(rune_home().resolve()),
           "RUNE_LOG_FILE": str(rune_logs() / "daemon.log")}
    if static := os.environ.get("RUNE_WEB_STATIC_DIR"):
        env["RUNE_WEB_STATIC_DIR"] = str(Path(static).resolve())
    return env


def _unit_quote(value: str, *, executable: bool = False) -> str:
    if any(char in value for char in "\n\r\0"):
        raise ValueError("Service paths and environment values must be single-line")
    if executable:
        value = value.replace("$", "$$")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def systemd_config(host: str = "127.0.0.1", port: int = 18789) -> str:
    from rune.daemon.endpoint import Endpoint
    endpoint = Endpoint(host, port)
    env = "\n".join(f"Environment={_unit_quote(f'{key}={value}')}"
                    for key, value in service_environment().items())
    return f"""[Unit]
Description=RUNE AI Daemon
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=exec
ExecStart={_unit_quote(sys.executable, executable=True)} -m rune.cli.main daemon run --host {_unit_quote(endpoint.host)} --port {endpoint.port}
Restart=on-failure
RestartSec=15
TimeoutStopSec=35
WorkingDirectory={_unit_quote(str(Path.home()))}
{env}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def _install_launchd(host: str, port: int) -> None:
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(plistlib.dumps(launchd_config(host, port)))

    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{_SERVICE_NAME}"],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
        check=True,
        capture_output=True,
    )
    log.info("launchd_service_installed", plist=str(plist_path))


def _uninstall_launchd() -> None:
    plist_path = _launchd_plist_path()

    if plist_path.exists():
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{_SERVICE_NAME}"],
            capture_output=True,
        )
        plist_path.unlink()
        log.info("launchd_service_uninstalled")
    else:
        log.warning("launchd_plist_not_found")


def _launchd_status() -> str:
    plist_path = _launchd_plist_path()
    if not plist_path.exists():
        return "not_installed"

    result = subprocess.run(
        ["launchctl", "list", _SERVICE_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and '"PID"' in result.stdout:
        return "running"
    return "stopped"


# Linux (systemd)

def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / _SYSTEMD_UNIT


def _install_systemd(host: str, port: int) -> None:
    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(systemd_config(host, port))

    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", _SYSTEMD_UNIT],
        check=True,
        capture_output=True,
    )
    log.info("systemd_service_installed", unit=str(unit_path))


def _uninstall_systemd() -> None:
    unit_path = _systemd_unit_path()

    if unit_path.exists():
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", _SYSTEMD_UNIT],
            capture_output=True,
        )
        unit_path.unlink()
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
        )
        log.info("systemd_service_uninstalled")
    else:
        log.warning("systemd_unit_not_found")


def _systemd_status() -> str:
    unit_path = _systemd_unit_path()
    if not unit_path.exists():
        return "not_installed"

    result = subprocess.run(
        ["systemctl", "--user", "is-active", _SYSTEMD_UNIT],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if status == "active":
        return "running"
    return "stopped"
