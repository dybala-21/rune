"""CLI subcommands for `rune advisor` — inspect advisor event persistence.

Provides a thin wrapper over ``MemoryStore.get_advisor_stats`` so users can
see whether the advisor layer is firing, which triggers are most common,
and whether injected plans correlate with successful episodes — without
writing SQL by hand.

Example:

    rune advisor stats                  # last 30 days, table format
    rune advisor stats --days 7
    rune advisor stats --format json
"""

from __future__ import annotations

import json as _json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

advisor_app = typer.Typer(help="Advisor layer inspection")
console = Console()


@advisor_app.callback()
def _advisor_callback() -> None:
    """Force typer to treat advisor_app as a multi-command group even
    when only one subcommand is registered. Without this, typer would
    interpret the single command as the app itself, breaking
    ``rune advisor stats`` invocation."""


def _render_table(stats: dict[str, Any]) -> None:
    """Pretty-print advisor stats as Rich tables."""
    since = stats.get("since_days", 30)
    total = stats.get("total_calls", 0)

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Window:", f"last {since} days")
    header.add_row("Total calls:", str(total))
    header.add_row(
        "Avg output tokens:",
        f"{stats.get('avg_output_tokens', 0):.1f}",
    )
    header.add_row(
        "Avg latency (ms):",
        f"{stats.get('avg_latency_ms', 0):.1f}",
    )
    console.print(header)
    console.print()

    if total == 0:
        console.print(
            "[dim]No advisor events recorded yet. Set "
            "RUNE_ADVISOR_MODEL to enable the advisor layer, then "
            "run an agent episode.[/dim]"
        )
        return

    by_trigger = stats.get("by_trigger") or {}
    if by_trigger:
        t = Table(title="By trigger", show_header=True, header_style="bold cyan")
        t.add_column("Trigger")
        t.add_column("Count", justify="right")
        t.add_column("Completion rate", justify="right")
        rates = stats.get("completion_rate_by_trigger") or {}
        for trig in sorted(by_trigger):
            rate = rates.get(trig)
            rate_str = f"{rate * 100:.1f}%" if rate is not None else "-"
            t.add_row(trig, str(by_trigger[trig]), rate_str)
        console.print(t)
        console.print()

    by_reason = stats.get("by_stuck_reason") or {}
    if by_reason:
        t = Table(
            title="Stuck sub-reasons",
            show_header=True,
            header_style="bold yellow",
        )
        t.add_column("Reason")
        t.add_column("Count", justify="right")
        for reason in sorted(by_reason):
            t.add_row(reason, str(by_reason[reason]))
        console.print(t)
        console.print()

    by_outcome = stats.get("by_outcome") or {}
    if by_outcome:
        t = Table(
            title="By episode outcome",
            show_header=True,
            header_style="bold magenta",
        )
        t.add_column("Outcome")
        t.add_column("Count", justify="right")
        for out in sorted(by_outcome):
            t.add_row(out, str(by_outcome[out]))
        console.print(t)


@advisor_app.command("stats")
def stats(
    days: int = typer.Option(
        30, "--days", "-d", help="Lookback window in days",
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or json",
    ),
) -> None:
    """Show aggregated advisor consultation metrics."""
    from rune.memory.store import get_memory_store

    store = get_memory_store()
    data = store.get_advisor_stats(since_days=days)

    if output_format.lower() == "json":
        console.print_json(_json.dumps(data))
        return

    _render_table(data)
