"""Operator commands for isolated Rune hosting."""

import asyncio
from pathlib import Path

import typer

from rune.cloud.store import CloudStore, cloud_home

cloud_app = typer.Typer(help="Experimental hosting: one Rune instance per owner (Linux/Incus)")


def store() -> CloudStore:
    return CloudStore(cloud_home())


@cloud_app.command("build-image")
def build_image(wheel: Path, web: Path, storage: str, base: str = "images:debian/13/cloud"):
    """Build a clean image from a local wheel and web/dist. Requires an initialized Incus host."""
    from rune.cloud.image import build_image as build
    from rune.cloud.incus import Incus
    typer.echo(asyncio.run(build(Incus(store()), wheel, web, storage, base)))


@cloud_app.command("provision")
def provision(owner: str, image: str, storage: str, cpus: int = 2, memory_gib: int = 4, disk_gib: int = 32):
    """Create the owner's VM, network and login token. Existing attempts are never overwritten."""
    from rune.cloud.incus import Incus
    token = asyncio.run(Incus(store()).provision(owner, image, storage=storage, cpus=cpus, memory_gib=memory_gib, disk_gib=disk_gib))
    typer.echo(f"Access token (shown once): {token}")


@cloud_app.command("register")
def register(owner: str, endpoint: str, token_file: Path):
    """Connect an existing isolated worker. The file contains its Rune API token."""
    typer.echo(f"Access token (shown once): {store().register(owner, endpoint, token_file.read_text().strip())}")


@cloud_app.command("disable")
def disable(owner: str):
    """Revoke access without deleting the owner's VM or files."""
    store().disable(owner)


@cloud_app.command("start")
def start(owner: str):
    from rune.cloud.incus import Incus
    asyncio.run(Incus(store()).set_running(owner, True))


@cloud_app.command("stop")
def stop(owner: str):
    from rune.cloud.incus import Incus
    asyncio.run(Incus(store()).set_running(owner, False))


@cloud_app.command("status")
def status():
    with store().connect() as db:
        for row in db.execute("SELECT owner, name, status FROM machines ORDER BY owner"):
            typer.echo(f"{row['owner']}: {row['name']} ({row['status']})")


@cloud_app.command("serve")
def serve(public_origin: str, host: str = "127.0.0.1", port: int = 18800):
    """Serve behind a TLS reverse proxy. public-origin is the exact HTTPS browser origin."""
    import uvicorn

    from rune.cloud.gateway import create_gateway
    uvicorn.run(create_gateway(store(), public_origin), host=host, port=port, access_log=False,
                ws_max_size=2 * 1024 * 1024, proxy_headers=False, limit_concurrency=200)
