"""Service installation for RUNE daemon.

Ported from src/daemon/service.ts - generates and installs launchd plist
(macOS) or systemd unit (Linux) for running the daemon as a system service.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_logs

log = get_logger(__name__)

_SERVICE_NAME = "com.rune.daemon"
_SYSTEMD_UNIT = "rune-daemon.service"


def install_service() -> None:
    """Install the RUNE daemon as a system service.

    On macOS: creates a launchd plist in ~/Library/LaunchAgents.
    On Linux: creates a systemd user unit in ~/.config/systemd/user.
    """
    system = platform.system()
    if system == "Darwin":
        _install_launchd()
    elif system == "Linux":
        _install_systemd()
    else:
        raise RuntimeError(f"Service installation not supported on {system}")


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


# macOS (launchd)

def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_SERVICE_NAME}.plist"


def _install_launchd() -> None:
    python = sys.executable
    log_dir = rune_logs()
    plist_path = _launchd_plist_path()

    plist_content = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>rune</string>
        <string>daemon</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir / "daemon-stdout.log"}</string>
    <key>StandardErrorPath</key>
    <string>{log_dir / "daemon-stderr.log"}</string>
    <key>WorkingDirectory</key>
    <string>{Path.home()}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}</string>
    </dict>
</dict>
</plist>
"""

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    subprocess.run(
        ["launchctl", "load", str(plist_path)],
        check=True,
        capture_output=True,
    )
    log.info("launchd_service_installed", plist=str(plist_path))


def _uninstall_launchd() -> None:
    plist_path = _launchd_plist_path()

    if plist_path.exists():
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
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
    if result.returncode == 0:
        return "running"
    return "stopped"


# Linux (systemd)

def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / _SYSTEMD_UNIT


def _install_systemd() -> None:
    python = sys.executable
    log_dir = rune_logs()
    unit_path = _systemd_unit_path()

    unit_content = f"""\
[Unit]
Description=RUNE AI Daemon
After=network.target

[Service]
Type=simple
ExecStart={python} -m rune daemon start
Restart=on-failure
RestartSec=5
WorkingDirectory={Path.home()}
Environment=PATH={os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
StandardOutput=append:{log_dir / "daemon-stdout.log"}
StandardError=append:{log_dir / "daemon-stderr.log"}

[Install]
WantedBy=default.target
"""

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_content)

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
