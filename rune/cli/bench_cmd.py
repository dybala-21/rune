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
from rune.bench.aa_score import DEFAULT_COMPONENT, format_score_text, score_paths
from rune.bench.audit import audit_attempt_dir
from rune.bench.runner import BenchRunOptions, run_bench_attempt
from rune.bench.summary import format_summary_csv, summarize_paths

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


@bench_app.command("aa-score")
def score_aa_results(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Harbor job/trial directories or RUNE attempt artifact directories"),
    ],
    component: Annotated[
        str,
        typer.Option("--component", "-c", help="AA component name or benchmark slug"),
    ] = DEFAULT_COMPONENT,
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or text"),
    ] = "json",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the score report to this path"),
    ] = None,
    allow_missing_fingerprint: Annotated[
        bool,
        typer.Option(
            "--allow-missing-fingerprint",
            help="Score exploratory runs without requiring fingerprint_gate.valid=true",
        ),
    ] = False,
) -> None:
    """Calculate an Artificial Analysis component score report."""
    if output_format not in {"json", "text"}:
        raise typer.BadParameter("--format must be 'json' or 'text'")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise typer.BadParameter(f"path does not exist: {missing[0]}")

    try:
        report = score_paths(
            paths,
            component=component,
            require_fingerprint=not allow_missing_fingerprint,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if output_format == "text":
        payload = format_score_text(report)
    else:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"

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
    agent_variant_id: Annotated[
        str | None,
        typer.Option("--agent-variant-id", help="Stable agent variant ID for provenance"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write artifacts without invoking the agent"),
    ] = False,
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Override agent max iterations for this attempt"),
    ] = None,
    timeout_seconds: Annotated[
        int | None,
        typer.Option("--timeout-seconds", help="Abort this attempt after N seconds"),
    ] = None,
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
    if max_steps is not None and max_steps < 1:
        raise typer.BadParameter("--max-steps must be >= 1")
    if timeout_seconds is not None and timeout_seconds < 1:
        raise typer.BadParameter("--timeout-seconds must be >= 1")

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
            agent_variant_id=agent_variant_id,
            dry_run=dry_run,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
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


@bench_app.command("summarize")
def summarize_bench_results(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Harbor job/trial directories or RUNE attempt artifact directories"),
    ],
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or csv"),
    ] = "json",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the summary to this path"),
    ] = None,
) -> None:
    """Summarize Harbor trial results and RUNE benchmark attempt artifacts."""
    if output_format not in {"json", "csv"}:
        raise typer.BadParameter("--format must be 'json' or 'csv'")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise typer.BadParameter(f"path does not exist: {missing[0]}")

    summary = summarize_paths(paths)
    if output_format == "csv":
        payload = format_summary_csv(summary)
    else:
        payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if output is None:
        typer.echo(payload, nl=False)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    typer.echo(f"Wrote {output}")
