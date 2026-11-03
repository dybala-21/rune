"""Operator commands for routing logins to isolated Rune workers.

Creating the per-owner VMs those workers run in is operated outside this repository.
"""

from pathlib import Path

import typer

from rune.cloud.store import CloudStore, cloud_home

cloud_app = typer.Typer(help="Remote access: route each owner's login to their own Rune worker")


def store() -> CloudStore:
    return CloudStore(cloud_home())


@cloud_app.command("register")
def register(owner: str, endpoint: str, token_file: Path):
    """Connect an existing isolated worker. The file contains its Rune API token."""
    typer.echo(f"Access token (shown once): {store().register(owner, endpoint, token_file.read_text().strip())}")


@cloud_app.command("disable")
def disable(owner: str):
    """Revoke access without deleting the owner's worker or files."""
    store().disable(owner)


@cloud_app.command("serve")
def serve(public_origin: str, host: str = "127.0.0.1", port: int = 18800):
    """Serve behind a TLS reverse proxy. public-origin is the exact HTTPS browser origin."""
    import uvicorn

    from rune.cloud.gateway import create_gateway
    uvicorn.run(create_gateway(store(), public_origin), host=host, port=port, access_log=False,
                ws_max_size=2 * 1024 * 1024, proxy_headers=False, limit_concurrency=200)
