"""Configure and run the optional credential broker."""

import os
from typing import Annotated

import typer

from rune.connectors.models import ConnectorPolicy
from rune.connectors.store import ConnectorStore, authority_key, broker_home

connector_app = typer.Typer(help="Credential-backed HTTP connectors")


@connector_app.command("add")
def add(name: str, origin: str, path_prefix: str = "/",
        method: Annotated[list[str] | None, typer.Option("--method")] = None,
        header: str = "Authorization", scheme: str = "Bearer"):
    """Store a connector policy and prompt for its credential without echoing it."""
    policy = ConnectorPolicy(name=name, origin=origin, path_prefix=path_prefix,
                             methods=method or ["GET"], header=header, scheme=scheme)
    store = ConnectorStore(broker_home())
    store.put(policy, typer.prompt("Credential", hide_input=True))
    authority_key(broker_home(), create=True)
    typer.echo(f"Saved connector {name}")


@connector_app.command("list")
def list_connectors():
    for policy in ConnectorStore(broker_home()).list():
        typer.echo(f"{policy['name']}: {','.join(policy['methods'])} {policy['origin']}{policy['path_prefix']}")


@connector_app.command("remove")
def remove(name: str):
    ConnectorStore(broker_home()).remove(name)
    typer.echo(f"Removed connector {name}")


@connector_app.command("serve")
def serve():
    """Run outside the workspace/container, under a separate supervised process."""
    import uvicorn
    from filelock import FileLock

    from rune.connectors.broker import create_broker

    root = broker_home()
    store = ConnectorStore(root)
    key = authority_key(root, create=True)
    with FileLock(root / "broker.lock", timeout=0):
        socket = root / "broker.sock"
        if socket.exists():
            if not socket.is_socket():
                raise ValueError("Broker socket path is occupied by another file")
            socket.unlink()
        previous = os.umask(0o077)
        try:
            uvicorn.run(create_broker(store, key), uds=str(socket), access_log=False)
        finally:
            os.umask(previous)
            if socket.is_socket():
                socket.unlink()
