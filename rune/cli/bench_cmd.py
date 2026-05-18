"""Benchmark helper commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rune.bench.aa_manifest import (
    TERMINAL_BENCH_V2_AA_TASKS,
    build_aa_attempt_matrix,
    build_artificial_analysis_manifest,
    validate_manifest,
)
from rune.bench.audit import audit_attempt_dir
from rune.bench.runner import BenchRunOptions, run_bench_attempt

bench_app = typer.Typer(help="Benchmark manifests and run helpers")


@bench_app.command("aa-manifest")
def write_aa_manifest(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the manifest JSON to this path"),
    ] = None,
) -> None:
    """Print or write the pinned Artificial Analysis Coding Agent manifest."""
    manifest = build_artificial_analysis_manifest()
    validate_manifest(manifest)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if output is None:
        typer.echo(payload, nl=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    typer.echo(f"Wrote {output}")


@bench_app.command("aa-matrix")
def write_aa_matrix(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the attempt matrix JSON to this path"),
    ] = None,
    component: Annotated[
        str | None,
        typer.Option("--component", "-c", help="Optional component name or benchmark slug filter"),
    ] = None,
) -> None:
    """Print or write the pinned 1074-attempt Artificial Analysis matrix."""
    matrix = build_aa_attempt_matrix()
    if component:
        key = component.lower()
        matrix = [
            row for row in matrix
            if row["component"].lower() == key or row["benchmark"].lower() == key
        ]
    payload = json.dumps(matrix, indent=2, sort_keys=True) + "\n"

    if output is None:
        typer.echo(payload, nl=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    typer.echo(f"Wrote {output}")


@bench_app.command("run")
def run_attempt(
    benchmark: Annotated[str, typer.Option("--benchmark", "-b", help="Benchmark name")] = "",
    task_id: Annotated[str, typer.Option("--task-id", "-t", help="Benchmark task ID")] = "",
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", "-i", help="Task instruction text"),
    ] = None,
    instruction_file: Annotated[
        Path | None,
        typer.Option("--instruction-file", help="Read task instruction from this file"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Artifact root directory"),
    ] = Path("runs/bench"),
    attempt_index: Annotated[
        int,
        typer.Option("--attempt-index", help="1-based repeat index for this task"),
    ] = 1,
    rune_home: Annotated[
        Path | None,
        typer.Option("--rune-home", help="Isolated RUNE_HOME for this attempt"),
    ] = None,
    cwd: Annotated[
        Path,
        typer.Option("--cwd", help="Workspace where the agent should run"),
    ] = Path("."),
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model override")] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider override"),
    ] = None,
    memory_mode: Annotated[
        str,
        typer.Option("--memory-mode", help="default or off"),
    ] = "default",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write artifacts without invoking the agent"),
    ] = False,
) -> None:
    """Run one local RUNE benchmark attempt and write structured artifacts."""
    if not benchmark:
        raise typer.BadParameter("--benchmark is required")
    if not task_id:
        raise typer.BadParameter("--task-id is required")
    if instruction and instruction_file:
        raise typer.BadParameter("Use either --instruction or --instruction-file, not both")
    if not instruction and not instruction_file:
        raise typer.BadParameter("One of --instruction or --instruction-file is required")
    if attempt_index < 1:
        raise typer.BadParameter("--attempt-index must be >= 1")
    if memory_mode not in {"default", "off"}:
        raise typer.BadParameter("--memory-mode must be 'default' or 'off'")

    loaded_instruction = (
        instruction_file.read_text(encoding="utf-8") if instruction_file is not None else instruction
    )
    assert loaded_instruction is not None

    resolved_output = output_dir.resolve()
    resolved_cwd = cwd.resolve()
    resolved_rune_home = (
        rune_home.resolve()
        if rune_home is not None
        else resolved_output / "_rune_home" / benchmark / task_id
    )

    attempt_dir = run_bench_attempt(
        BenchRunOptions(
            benchmark=benchmark,
            task_id=task_id,
            instruction=loaded_instruction,
            output_dir=resolved_output,
            rune_home=resolved_rune_home,
            cwd=resolved_cwd,
            attempt_index=attempt_index,
            model=model,
            provider=provider,
            memory_mode=memory_mode,
            dry_run=dry_run,
        )
    )
    typer.echo(f"Wrote {attempt_dir}")


@bench_app.command("terminal-smoke")
def terminal_smoke(
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Number of AA Terminal-Bench tasks to print"),
    ] = 10,
    harbor_command: Annotated[
        bool,
        typer.Option("--harbor-command", help="Print Harbor commands for each task"),
    ] = False,
) -> None:
    """Print the first N Terminal-Bench tasks from the AA84 subset."""
    if count < 1:
        raise typer.BadParameter("--count must be >= 1")
    tasks = TERMINAL_BENCH_V2_AA_TASKS[:count]
    if harbor_command:
        for task in tasks:
            typer.echo(
                "harbor run "
                "-d terminal-bench@2.0 "
                f"--include-task-name {task} "
                f"--agent-env RUNE_HARBOR_TASK_ID={task} "
                "--agent-import-path benchmarks.harbor.rune_agent:RuneInstalledAgent"
            )
        return
    for task in tasks:
        typer.echo(task)


@bench_app.command("audit")
def audit_attempt(
    attempt_dir: Annotated[Path, typer.Argument(help="Benchmark attempt artifact directory")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write audit JSON to this path"),
    ] = None,
) -> None:
    """Audit one benchmark attempt for leakage and benchmark-rule violations."""
    if not attempt_dir.exists():
        raise typer.BadParameter(f"attempt directory does not exist: {attempt_dir}")
    result = audit_attempt_dir(attempt_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(payload, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    typer.echo(f"Wrote {output}")
