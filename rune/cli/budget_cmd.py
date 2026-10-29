"""Inspect unresolved usage and record confirmed provider charges."""

import json
from typing import Annotated

import typer

from rune.agent.spend_budget import SpendLedger

budget_app = typer.Typer(help="Shared model spending and unresolved usage")


@budget_app.command("status")
def status() -> None:
    from rune.config.loader import get_config

    runtime = get_config().runtime
    ledger = SpendLedger()
    ledger.recover()
    typer.echo(json.dumps({"dailyLimitUsd": runtime.daily_cost_usd,
                           "monthlyLimitUsd": runtime.monthly_cost_usd,
                           "unresolved": ledger.unresolved()}, indent=2))


@budget_app.command("reconcile")
def reconcile(request_id: str,
              cost_usd: Annotated[float, typer.Option("--cost-usd", min=0)],
              evidence: Annotated[str, typer.Option("--evidence", help="Reference to confirmed provider usage or invoice")]) -> None:
    try:
        SpendLedger().reconcile(request_id, cost_usd, evidence)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("Usage reconciled. The confirmed cost remains charged to its original budget period.")
