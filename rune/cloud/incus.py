"""Provision persistent VMs on an operator-managed, local Incus host."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import shutil

import httpx
from filelock import FileLock

from rune.cloud.store import CloudStore, owner_key


class Incus:
    def __init__(self, store: CloudStore):
        self.store = store

    async def run(self, *args: str, stdin: bytes | None = None, timeout: int = 120) -> str:
        if not shutil.which("incus"):
            raise RuntimeError("Incus is required on the Linux VM host; no container fallback is used")
        process = await asyncio.create_subprocess_exec(
            "incus", *args, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            # stdin and Incus errors can contain provisioned credentials.
            raise RuntimeError(f"Incus {args[0]} failed (exit {process.returncode}); inspect the instance on the host")
        return stdout.decode()

    async def owned(self, owner: str, name: str) -> dict:
        # An empty result differs from a daemon/permission failure; never create on errors.
        instances = json.loads(await self.run("list", f"^{name}$", "--format=json"))
        if len(instances) != 1:
            raise ValueError("Provisioned instance is missing or ambiguous")
        item = instances[0]
        if item.get("name") != name or item.get("type") != "virtual-machine" or item.get("config", {}).get("user.rune.owner") != owner_key(owner):
            raise ValueError("Instance ownership or VM type does not match")
        return item

    @staticmethod
    def addresses(slot: int) -> tuple[str, str]:
        base = int(ipaddress.IPv4Address("10.207.0.0")) + slot * 4
        return str(ipaddress.IPv4Address(base + 1)), str(ipaddress.IPv4Address(base + 2))

    async def provision(self, owner: str, image: str, *, storage: str, cpus: int = 2, memory_gib: int = 4, disk_gib: int = 32) -> str:
        if not image or image.startswith("-") or not storage or storage.startswith("-"):
            raise ValueError("An operator-prepared Rune VM image and storage pool are required")
        if not 1 <= cpus <= 64 or not 2 <= memory_gib <= 256 or not 8 <= disk_gib <= 2048:
            raise ValueError("VM resource limits are outside the supported range")
        with FileLock(self.store.root / "provision.lock", timeout=0):
            machine = self.store.reserve(owner)
            if machine["status"] != "reserved":
                raise ValueError("This owner already has a provision attempt. Inspect/start the existing VM instead of recreating it")
            name = machine["name"]
            if json.loads(await self.run("list", f"^{name}$", "--format=json")):
                raise ValueError("Instance name already exists; no existing VM was changed")
            self.store.status(owner, "provisioning")
            gateway, address = self.addresses(machine["slot"])
            network = "rn-" + owner_key(owner)[:10]
            acl = f"{network}-acl"
            try:
                await self.run("network", "acl", "create", acl, f"user.rune.owner={owner_key(owner)}")
                await self.run("network", "acl", "rule", "add", acl, "egress", "action=reject",
                               "destination=0.0.0.0/8,10.0.0.0/8,100.64.0.0/10,127.0.0.0/8,169.254.0.0/16,172.16.0.0/12,192.168.0.0/16,224.0.0.0/4,240.0.0.0/4")
                await self.run("network", "acl", "rule", "add", acl, "egress", "action=reject", "destination=::/0")
                await self.run("network", "acl", "rule", "add", acl, "egress", "action=allow", "protocol=tcp", "destination_port=80,443")
                await self.run("network", "acl", "rule", "add", acl, "ingress", "action=allow", f"source={gateway}/32", "protocol=tcp", "destination_port=18789")
                # One bridge per owner avoids the intra-bridge ACL limitation.
                await self.run("network", "create", network, f"ipv4.address={gateway}/30", "ipv4.nat=true", "ipv6.address=none",
                               f"security.acls={acl}", f"user.rune.owner={owner_key(owner)}")
                await self.run("init", image, name, "--vm", "--no-profiles", "--storage", storage,
                               "--network", network, "-c", f"user.rune.owner={owner_key(owner)}",
                               "-c", f"limits.cpu={cpus}", "-c", f"limits.memory={memory_gib}GiB",
                               "-c", "boot.autostart=true", "-d", f"root,size={disk_gib}GiB")
                await self.run("config", "device", "set", name, "eth0", f"ipv4.address={address}", "security.ipv4_filtering=true", "security.mac_filtering=true")
                await self.owned(owner, name)
                await self.run("start", name)
                await self.wait_ready(name)
                token = await self.run("exec", name, "--", "/opt/rune/bin/python", "-m", "rune.cloud.guest", "initialize")
                token = token.strip()
                await self.run("exec", name, "--", "systemctl", "daemon-reload")
                await self.run("exec", name, "--", "systemctl", "enable", "--now", "rune-broker", "rune-worker")
                await self.wait_http(f"http://{address}:18789", token)
                # Registration is last. A partially configured VM never receives user traffic.
                login = self.store.register(owner, f"http://{address}:18789", token)
                self.store.status(owner, "running")
                return login
            except BaseException:
                self.store.status(owner, "failed")
                raise

    async def wait_ready(self, name: str, timeout: int = 180) -> None:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    await self.run("exec", name, "--", "test", "-x", "/opt/rune/bin/python", timeout=10)
                    return
                except RuntimeError:
                    await asyncio.sleep(2)

    async def wait_agent(self, name: str, timeout: int = 180) -> None:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    await self.run("exec", name, "--", "true", timeout=10)
                    return
                except RuntimeError:
                    await asyncio.sleep(2)

    async def wait_http(self, endpoint: str, token: str, timeout: int = 90) -> None:
        async with asyncio.timeout(timeout), httpx.AsyncClient(trust_env=False, timeout=3) as client:
            while True:
                try:
                    url = endpoint + "/api/runs/snapshot?sessionId=cloud-readiness-probe"
                    public = await client.get(url)
                    if public.status_code not in {401, 403}:
                        raise ValueError("Worker does not enforce authentication")
                    response = await client.get(url, headers={"authorization": f"Bearer {token}"})
                    if response.status_code == 200 and "run" in response.json():
                        return
                except httpx.HTTPError:
                    # The service may still be starting after the VM agent becomes available.
                    await asyncio.sleep(1)
                    continue
                await asyncio.sleep(1)

    async def set_running(self, owner: str, running: bool) -> None:
        with FileLock(self.store.root / "provision.lock", timeout=0):
            with self.store.connect() as db:
                row = db.execute("SELECT * FROM machines WHERE owner=?", (owner,)).fetchone()
            if row is None:
                raise ValueError("Owner has no provisioned VM")
            instance = await self.owned(owner, row["name"])
            self.store.status(owner, "starting" if running else "stopping")
            if not running:
                self.store.revoke_sessions(owner)
            if instance.get("status") != ("Running" if running else "Stopped"):
                await self.run("start" if running else "stop", row["name"])
            if running:
                await self.wait_ready(row["name"])
                with self.store.connect() as db:
                    worker = db.execute("SELECT endpoint, upstream_token FROM workers WHERE owner=?", (owner,)).fetchone()
                if worker is None:
                    raise ValueError("VM has no completed registration; inspect the failed provision attempt")
                await self.wait_http(worker["endpoint"], worker["upstream_token"])
            self.store.status(owner, "running" if running else "stopped")
