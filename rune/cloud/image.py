"""Build a credential-free VM image from a local wheel and compiled web assets."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rune.cloud.incus import Incus


async def build_image(driver: Incus, wheel: Path, web: Path, storage: str,
                      base: str = "images:debian/13/cloud") -> str:
    wheel, web = wheel.resolve(strict=True), web.resolve(strict=True)
    if not wheel.name.endswith(".whl") or not (web / "index.html").is_file():
        raise ValueError("Provide a Rune wheel and the compiled web/dist directory")
    if any(p.is_symlink() for p in web.rglob("*")):
        raise ValueError("Web assets must not contain symlinks")
    name = "rune-image-" + uuid4().hex[:12]
    await driver.run("launch", base, name, "--vm", "--storage", storage,
                     "-c", "limits.cpu=2", "-c", "limits.memory=4GiB", "-d", "root,size=32GiB",
                     "-c", f"user.rune.builder={name}")
    # Keep a failed builder for inspection; never remove a potentially useful disk automatically.
    await driver.wait_agent(name)
    await driver.run("exec", name, "--", "cloud-init", "status", "--wait", timeout=300)
    await driver.run("exec", name, "--", "sh", "-ec",
                     "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv docker.io && python3 -m venv /opt/rune", timeout=600)
    await driver.run("file", "push", str(wheel), f"{name}/tmp/{wheel.name}")
    await driver.run("exec", name, "--", "/opt/rune/bin/pip", "install", f"/tmp/{wheel.name}", "playwright>=1.58.0", timeout=600)
    await driver.run("exec", name, "--", "env", "PLAYWRIGHT_BROWSERS_PATH=/opt/rune-browsers", "/opt/rune/bin/python", "-m", "playwright", "install", "--with-deps", "chromium", timeout=600)
    await driver.run("exec", name, "--", "useradd", "--create-home", "--user-group", "--groups", "docker", "rune")
    await driver.run("exec", name, "--", "mkdir", "-p", "/opt/rune-web")
    for path in sorted(web.iterdir()):
        await driver.run("file", "push", "--recursive", str(path), f"{name}/opt/rune-web/")
    await driver.run("exec", name, "--", "systemctl", "enable", "--now", "docker")
    await driver.run("exec", name, "--", "docker", "pull", "python:3.13-slim", timeout=300)
    await driver.run("exec", name, "--", "rm", "--", f"/tmp/{wheel.name}")
    await driver.run("exec", name, "--", "cloud-init", "clean", "--logs", "--machine-id")
    await driver.run("stop", name)
    alias = "rune-vm-" + uuid4().hex[:12]
    await driver.run("publish", name, "--alias", alias, timeout=600)
    import json
    instances = json.loads(await driver.run("list", f"^{name}$", "--format=json"))
    if len(instances) == 1 and instances[0].get("config", {}).get("user.rune.builder") == name:
        await driver.run("delete", name)
    return alias
