"""Entrypoints installed in the trusted VM image, never in an agent workspace."""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from pathlib import Path

WORKSPACE = "/workspace/projects"
HOME = "/var/lib/rune"
BROKER = "/var/lib/rune-broker"


def initialize() -> str:
    """Create an empty owner's private state. Refuse to overwrite an existing identity."""
    from rune.connectors.store import ConnectorStore, authority_key, private_directory

    private_directory(Path(HOME))
    ConnectorStore(Path(BROKER))
    authority_key(Path(BROKER), create=True)
    Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
    import pwd
    account = pwd.getpwnam("rune")
    token = "rune_" + secrets.token_urlsafe(32)
    path = Path(HOME) / "api-tokens.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
        json.dump({token: {"created_at": time.time(), "label": "cloud-gateway"}}, stream)
    for root in (Path(HOME), Path(BROKER), Path(WORKSPACE)):
        os.chown(root, account.pw_uid, account.pw_gid)
        for child in root.rglob("*"):
            os.chown(child, account.pw_uid, account.pw_gid, follow_symlinks=False)
    for name, command in {"rune-worker": "/opt/rune/bin/python -m rune.cloud.guest worker",
                          "rune-broker": "/opt/rune/bin/rune connector serve"}.items():
        unit = f"""[Unit]
Description={name}
After=network-online.target docker.service
Wants=network-online.target
[Service]
Type=simple
User=rune
Group=rune
SupplementaryGroups=docker
WorkingDirectory={WORKSPACE}
Environment=RUNE_HOME={HOME}
Environment=RUNE_BROKER_HOME={BROKER}
ExecStart={command}
Restart=on-failure
RestartSec=3
UMask=0077
[Install]
WantedBy=multi-user.target
"""
        Path(f"/etc/systemd/system/{name}.service").write_text(unit)
    return token


def worker() -> None:
    os.environ.update(RUNE_HOME=HOME, RUNE_BROKER_HOME=BROKER, RUNE_WORKSPACE=WORKSPACE,
                      RUNE_ISOLATION_ROOT=WORKSPACE, RUNE_CLOUD_WORKER="1", RUNE_REQUIRE_TOKEN="1",
                      RUNE_APPROVAL_MODE="standard", RUNE_WEB_STATIC_DIR="/opt/rune-web",
                      PLAYWRIGHT_BROWSERS_PATH="/opt/rune-browsers",
                      RUNE_TERMINAL_ENABLED="0")
    os.chdir(WORKSPACE)
    from rune.cli.main import web
    web(host="0.0.0.0", port=18789, no_open=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["initialize"]:
        print(initialize())
    elif sys.argv[1:] == ["worker"]:
        worker()
    else:
        raise SystemExit("Expected initialize or worker")
