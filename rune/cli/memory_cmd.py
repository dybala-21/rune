"""CLI subcommands for `rune memory`."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

memory_app = typer.Typer(help="Memory inspection and management")
console = Console()


def _mem_dir() -> Path:
    from rune.memory.markdown_store import memory_dir
    return memory_dir()


@memory_app.command("show")
def show(
    learned: bool = typer.Option(False, "--learned", help="Show learned.md"),
    daily: str | None = typer.Option(None, "--daily", help="Show daily log (YYYY-MM-DD or 'today')"),
    profile: bool = typer.Option(False, "--profile", help="Show user profile"),
) -> None:
    """Show memory file contents."""
    d = _mem_dir()

    if learned:
        path = d / "learned.md"
    elif daily is not None:
        if daily == "today" or daily == "":
            from datetime import UTC, datetime
            daily = datetime.now(UTC).strftime("%Y-%m-%d")
        path = d / "daily" / f"{daily}.md"
    elif profile:
        path = d / "user-profile.md"
    else:
        path = d / "MEMORY.md"

    if not path.exists():
        console.print(f"[dim]File not found: {path}[/dim]")
        raise typer.Exit(1)

    console.print(path.read_text(encoding="utf-8"))


@memory_app.command("edit")
def edit(
    learned: bool = typer.Option(False, "--learned", help="Edit learned.md"),
    rules: bool = typer.Option(False, "--rules", help="Edit project rules.md"),
    profile: bool = typer.Option(False, "--profile", help="Edit user profile"),
) -> None:
    """Open a memory file in $EDITOR."""
    d = _mem_dir()

    if learned:
        path = d / "learned.md"
    elif rules:
        path = Path.cwd() / ".rune" / "memory" / "rules.md"
    elif profile:
        path = d / "user-profile.md"
    else:
        path = d / "MEMORY.md"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(path)], check=False)


@memory_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    explain: bool = typer.Option(False, "--explain", help="Show scoring breakdown"),
) -> None:
    """Search all memory with scores and sources."""
    import asyncio

    from rune.memory.manager import MemoryManager

    async def _search() -> None:
        mgr = MemoryManager()
        await mgr.initialize()
        results = await mgr.search(query, k=10)

        if not results:
            console.print("[dim]No results found.[/dim]")
            return

        table = Table(title=f"Results for '{query}'")
        table.add_column("Score", width=6)
        table.add_column("Source", width=25)
        table.add_column("Content", ratio=1)

        for r in results:
            score = f"{r.score:.3f}"
            source = r.metadata.type
            if hasattr(r.metadata, "source_file"):
                source = getattr(r.metadata, "source_file", source)
            content = (r.metadata.summary or r.text or r.id)[:80]
            table.add_row(score, source, content)

        console.print(table)

    asyncio.run(_search())


@memory_app.command("promote")
def promote(
    key: str = typer.Argument(..., help="Fact key to promote from learned.md to MEMORY.md"),
) -> None:
    """Move a fact from learned.md to MEMORY.md."""
    from rune.memory.markdown_store import (
        append_to_memory_md,
        parse_learned_md,
        remove_learned_fact,
    )

    facts = parse_learned_md()
    fact = next((f for f in facts if f["key"].lower() == key.lower()), None)
    if fact is None:
        console.print(f"[red]Key '{key}' not found in learned.md[/red]")
        raise typer.Exit(1)

    section = {
        "preference": "Preferences",
        "environment": "Environment",
        "lesson": "Notes",
        "decision": "Notes",
    }.get(fact["category"], "Notes")

    append_to_memory_md(section, f"{fact['key']}: {fact['value']}")
    remove_learned_fact(key)
    console.print(f"Promoted '{key}' to MEMORY.md # {section}")


@memory_app.command("forget")
def forget(
    key: str = typer.Argument(..., help="Fact key to delete and suppress"),
) -> None:
    """Delete a fact and prevent re-extraction."""
    from rune.memory.markdown_store import learned_md_has_key, remove_learned_fact
    from rune.memory.state import suppress_fact

    value = learned_md_has_key(key)
    if value is None:
        console.print(f"[red]Key '{key}' not found in learned.md[/red]")
        raise typer.Exit(1)

    remove_learned_fact(key)
    suppress_fact(key, value, reason="user_deleted")
    console.print(f"Removed '{key}' and suppressed future extraction.")


@memory_app.command("unsuppress")
def unsuppress(
    key: str = typer.Argument(..., help="Fact key to allow re-extraction"),
) -> None:
    """Remove a key from the suppression list."""
    from rune.memory.state import unsuppress_fact

    if unsuppress_fact(key):
        console.print(f"Unsuppressed '{key}'.")
    else:
        console.print(f"[dim]'{key}' was not suppressed.[/dim]")


@memory_app.command("conflicts")
def conflicts() -> None:
    """Show recent fact conflicts."""
    from rune.memory.state import load_conflicts

    data = load_conflicts()
    if not data:
        console.print("[dim]No conflicts recorded.[/dim]")
        return

    table = Table(title="Fact Conflicts")
    table.add_column("Key")
    table.add_column("Old Value")
    table.add_column("New Value")
    table.add_column("Resolved At")

    for c in data[-20:]:
        table.add_row(c["key"], c["old_value"], c["new_value"], c.get("resolved_at", ""))

    console.print(table)


@memory_app.command("stats")
def stats() -> None:
    """Show memory statistics."""
    from rune.memory.markdown_store import parse_learned_md, parse_memory_md
    from rune.memory.state import load_fact_meta

    memory_facts = parse_memory_md()
    total_memory = sum(len(v) for v in memory_facts.values())
    learned = parse_learned_md()
    meta = load_fact_meta()

    console.print(f"MEMORY.md: {total_memory} facts across {len(memory_facts)} sections")
    console.print(f"learned.md: {len(learned)} facts")

    # Most used
    by_hits = sorted(meta.items(), key=lambda x: x[1].get("hit_count", 0), reverse=True)
    if by_hits:
        console.print("\nMost used:")
        for key, m in by_hits[:5]:
            console.print(f"  {key}: {m.get('hit_count', 0)} hits")

    # Never used
    never_used = [k for k, m in meta.items() if m.get("hit_count", 0) == 0]
    if never_used:
        console.print(f"\nNever used: {len(never_used)} facts")

    # GC candidates
    gc_candidates = [k for k, m in meta.items() if m.get("confidence", 1.0) < 0.3]
    if gc_candidates:
        console.print(f"GC candidates (confidence < 0.3): {len(gc_candidates)}")

    # Daily logs
    daily_dir = _mem_dir() / "daily"
    if daily_dir.exists():
        daily_count = len(list(daily_dir.glob("*.md")))
        console.print(f"\nDaily logs: {daily_count} files")


@memory_app.command("gc")
def gc() -> None:
    """Prune low-confidence facts and compact old daily logs."""
    from rune.memory.markdown_store import prune_learned_md
    from rune.memory.state import load_fact_meta, save_fact_meta, suppress_fact

    # Prune learned.md over soft cap
    removed = prune_learned_md()
    for key in removed:
        suppress_fact(key, "", reason="auto_pruned")
    if removed:
        console.print(f"Pruned {len(removed)} low-confidence facts from learned.md")

    # Decay confidence for unused facts
    meta = load_fact_meta()
    decayed = 0
    from datetime import UTC, datetime
    now = datetime.now(UTC)

    for _key, m in list(meta.items()):
        if m.get("hit_count", 0) == 0:
            created = m.get("created", "")
            if created:
                try:
                    age = (now - datetime.fromisoformat(created)).days
                    if age > 30:
                        m["confidence"] = m.get("confidence", 0.5) * 0.9
                        decayed += 1
                except (ValueError, TypeError):
                    pass

    if decayed:
        save_fact_meta(meta)
        console.print(f"Decayed confidence for {decayed} unused facts")

    if not removed and not decayed:
        console.print("[dim]Nothing to clean up.[/dim]")


@memory_app.command("rebuild")
def rebuild() -> None:
    """Full FAISS reindex from markdown files."""
    import asyncio

    from rune.memory.markdown_indexer import full_rebuild

    async def _rebuild() -> None:
        stats = await full_rebuild()
        console.print(f"Rebuilt index: {stats.get('added', 0)} chunks indexed")

    asyncio.run(_rebuild())


@memory_app.command("validate")
def validate() -> None:
    """Check markdown files for parse errors."""
    from rune.memory.markdown_store import parse_learned_md, parse_memory_md, parse_user_profile

    errors = 0

    try:
        parse_memory_md()
        console.print("[green]MEMORY.md: OK[/green]")
    except Exception as e:
        console.print(f"[red]MEMORY.md: {e}[/red]")
        errors += 1

    try:
        facts = parse_learned_md()
        broken = [f for f in facts if f["category"] == "general" and f["confidence"] == 0.3]
        if broken:
            console.print(f"[yellow]learned.md: {len(broken)} lines with broken format (parsed as free-text)[/yellow]")
        else:
            console.print("[green]learned.md: OK[/green]")
    except Exception as e:
        console.print(f"[red]learned.md: {e}[/red]")
        errors += 1

    try:
        parse_user_profile()
        console.print("[green]user-profile.md: OK[/green]")
    except Exception as e:
        console.print(f"[red]user-profile.md: {e}[/red]")
        errors += 1

    if errors:
        raise typer.Exit(1)


@memory_app.command("export")
def export(
    json_output: bool = typer.Option(True, "--json", help="Export as JSON"),
) -> None:
    """Export all memory as JSON."""
    from rune.memory.markdown_store import parse_learned_md, parse_memory_md, parse_user_profile
    from rune.memory.state import load_conflicts, load_fact_meta, load_suppressed

    data = {
        "memory_md": parse_memory_md(),
        "learned": parse_learned_md(),
        "profile": parse_user_profile(),
        "fact_meta": load_fact_meta(),
        "suppressed": load_suppressed(),
        "conflicts": load_conflicts(),
    }

    console.print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


@memory_app.command("restore")
def restore() -> None:
    """Restore MEMORY.md or learned.md from backup."""
    d = _mem_dir()
    state = d / ".state"

    restored = 0
    for name in ("MEMORY.md", "learned.md"):
        bak = state / f"{name}.bak"
        if bak.exists():
            target = d / name
            target.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(f"Restored {name} from backup")
            restored += 1

    if not restored:
        console.print("[dim]No backups found in .state/[/dim]")
