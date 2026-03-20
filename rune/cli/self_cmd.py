"""``rune self`` - update, uninstall, and status commands.

Detects the installation method (uv / pipx / pip) from ``sys.executable``
and delegates to the appropriate package manager.

Designed to be a **thin, fast** module - avoids importing heavy rune
internals so that the running process can exit cleanly after the
underlying venv is replaced by the upgrade.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

self_app = typer.Typer(help="Manage the RUNE installation")
console = Console(stderr=True)

PACKAGE_NAME = "rune-ai"


# Installer detection

def _detect_installer() -> str:
    """Detect how rune-ai was installed by inspecting sys.executable path.

    Returns one of: "uv", "pipx", "pip", "source".

    Priority: uv > pipx > source > pip.
    uv and pipx are detected by their canonical venv paths.
    Source is detected by the presence of a .git directory alongside
    pyproject.toml (a PyPI install won't have .git).
    """
    exe_resolved = str(Path(sys.executable).resolve())
    if "/uv/tools/" in exe_resolved:
        return "uv"
    if "/pipx/venvs/" in exe_resolved:
        return "pipx"
    # Source checkout: pyproject.toml + either .git or local .venv must exist.
    # A PyPI wheel install won't have these alongside the rune package.
    project_root = Path(__file__).resolve().parents[2]
    if (project_root / "pyproject.toml").exists() and (
        (project_root / ".git").exists() or (project_root / ".venv").exists()
    ):
        return "source"
    return "pip"


def _check_daemon() -> int | None:
    """Return daemon PID if running, else None. Minimal import."""
    try:
        from rune.daemon.process_lock import get_lock_owner
        return get_lock_owner()
    except Exception:
        return None


def _stop_daemon() -> bool:
    """Attempt to stop the daemon gracefully. Returns True on success."""
    try:
        from rune.daemon.client import DaemonClient
        client = DaemonClient()
        import asyncio
        asyncio.run(client.shutdown())
        return True
    except Exception:
        return False


# Commands

@self_app.command()
def update(
    force: bool = typer.Option(False, "--force", "-f", help="Update even if daemon is running"),
) -> None:
    """Update RUNE to the latest version."""
    installer = _detect_installer()
    console.print(f"  Installer: [cyan]{installer}[/cyan]")

    if installer == "source":
        console.print("[yellow]Installed from source — use 'git pull && uv sync' instead.[/yellow]")
        raise typer.Exit(1)

    # Check daemon
    daemon_pid = _check_daemon()
    if daemon_pid and not force:
        console.print(f"[yellow]Daemon is running (PID {daemon_pid}).[/yellow]")
        console.print("  Stop it first:  [bold]rune daemon stop[/bold]")
        console.print("  Or use:         [bold]rune self update --force[/bold]")
        raise typer.Exit(1)

    if daemon_pid and force:
        console.print(f"[yellow]Stopping daemon (PID {daemon_pid})...[/yellow]")
        _stop_daemon()

    # Run upgrade
    console.print("[blue]==> Updating RUNE...[/blue]")

    if installer == "uv":
        if not shutil.which("uv"):
            console.print("[red]error:[/red] uv not found in PATH")
            raise typer.Exit(1)
        result = subprocess.run(["uv", "tool", "upgrade", PACKAGE_NAME])
    elif installer == "pipx":
        if not shutil.which("pipx"):
            console.print("[red]error:[/red] pipx not found in PATH")
            raise typer.Exit(1)
        result = subprocess.run(["pipx", "upgrade", PACKAGE_NAME])
    else:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME])

    if result.returncode != 0:
        console.print("[red]Update failed.[/red]")
        raise typer.Exit(result.returncode)

    console.print("[green]==> RUNE updated successfully![/green]")

    if daemon_pid:
        console.print("  Restart daemon: [bold]rune daemon start[/bold]")

    # Exit immediately - don't use any more rune.* imports after venv replacement
    raise typer.Exit(0)


@self_app.command()
def uninstall(
    force: bool = typer.Option(False, "--force", "-f", help="Uninstall even if daemon is running"),
    keep_data: bool = typer.Option(True, "--keep-data/--remove-data", help="Keep ~/.rune/ data directory"),
) -> None:
    """Remove RUNE from your system."""
    installer = _detect_installer()
    console.print(f"  Installer: [cyan]{installer}[/cyan]")

    if installer == "source":
        console.print("[yellow]Installed from source — just delete the project directory.[/yellow]")
        raise typer.Exit(1)

    # Check daemon
    daemon_pid = _check_daemon()
    if daemon_pid and not force:
        console.print(f"[yellow]Daemon is running (PID {daemon_pid}).[/yellow]")
        console.print("  Stop it first:  [bold]rune daemon stop[/bold]")
        console.print("  Or use:         [bold]rune self uninstall --force[/bold]")
        raise typer.Exit(1)

    if daemon_pid:
        console.print(f"[yellow]Stopping daemon (PID {daemon_pid})...[/yellow]")
        _stop_daemon()

    # Confirm
    if sys.stdin.isatty():
        confirm = input("Uninstall RUNE? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Uninstall
    console.print("[blue]==> Uninstalling RUNE...[/blue]")

    if installer == "uv":
        result = subprocess.run(["uv", "tool", "uninstall", PACKAGE_NAME])
    elif installer == "pipx":
        result = subprocess.run(["pipx", "uninstall", PACKAGE_NAME])
    else:
        result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", PACKAGE_NAME])

    if result.returncode != 0:
        console.print("[red]Uninstall failed.[/red]")
        raise typer.Exit(result.returncode)

    # Data cleanup
    rune_home = Path.home() / ".rune"
    if not keep_data and rune_home.exists():
        console.print(f"[yellow]Removing {rune_home}...[/yellow]")
        import shutil as _shutil
        _shutil.rmtree(rune_home, ignore_errors=True)
        console.print("[green]Data removed.[/green]")

    console.print("[green]==> RUNE uninstalled.[/green]")
    if keep_data and rune_home.exists():
        console.print(f"  Your data is still at: [dim]{rune_home}[/dim]")
        console.print(f"  To remove it: [bold]rm -rf {rune_home}[/bold]")

    raise typer.Exit(0)


@self_app.command()
def status() -> None:
    """Show installation info."""
    from rune import __version__

    installer = _detect_installer()
    exe = sys.executable
    rune_bin = shutil.which("rune") or "(not on PATH)"
    rune_home = Path.home() / ".rune"

    daemon_pid = _check_daemon()
    daemon_status = f"running (PID {daemon_pid})" if daemon_pid else "stopped"

    console.print(f"  Version:   [bold]{__version__}[/bold]")
    console.print(f"  Installer: [cyan]{installer}[/cyan]")
    console.print(f"  Binary:    {rune_bin}")
    console.print(f"  Python:    {exe}")
    console.print(f"  Data dir:  {rune_home}  ({'exists' if rune_home.exists() else 'not created yet'})")
    console.print(f"  Daemon:    {daemon_status}")
