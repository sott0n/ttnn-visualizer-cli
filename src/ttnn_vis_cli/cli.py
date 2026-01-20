"""Main CLI entry point for TTNN Visualizer CLI."""

from pathlib import Path

import click

from .commands import devices, info, l1, memory, operations, perf, sharding, tensors
from .output.formatter import OutputFormat


def get_format(ctx: click.Context) -> OutputFormat:
    """Get output format from context."""
    format_str = ctx.obj.get("format", "table")
    return OutputFormat(format_str)


@click.group()
@click.version_option(version="0.1.0", prog_name="ttnn-vis-cli")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format (table, json, csv)",
)
@click.pass_context
def cli(ctx: click.Context, format: str) -> None:
    """TTNN Visualizer CLI - CLI tool for TTNN profiling visualization and analysis.

    Analyze TTNN profiling reports including operations, tensors,
    memory usage, and performance metrics.
    """
    ctx.ensure_object(dict)
    ctx.obj["format"] = format


# Register commands
cli.add_command(info.info)
cli.add_command(devices.devices)
cli.add_command(operations.operations)
cli.add_command(operations.operation)
cli.add_command(tensors.tensors)
cli.add_command(tensors.tensor)
cli.add_command(memory.memory)
cli.add_command(memory.buffers)
cli.add_command(perf.perf)
cli.add_command(l1.l1_report, name="l1-report")
cli.add_command(sharding.sharding)


@cli.command()
@click.option(
    "--profiler",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to profiler SQLite database (db.sqlite)",
)
@click.option(
    "--performance",
    "-P",
    type=click.Path(exists=True, path_type=Path),
    help="Path to performance report directory or CSV file",
)
def tui(profiler: Path | None, performance: Path | None) -> None:
    """Launch interactive TUI for browsing profiling data.

    Example:
        ttnn-vis-cli tui --profiler samples/profiler/db.sqlite
        ttnn-vis-cli tui -p samples/profiler/db.sqlite -P samples/performance
    """
    try:
        from .tui import run_tui
    except ImportError:
        click.echo(
            "Error: TUI dependencies not installed.\n"
            "Install with: pip install ttnn-vis-cli[tui]",
            err=True,
        )
        raise SystemExit(1)

    if not profiler and not performance:
        click.echo(
            "Error: At least one of --profiler or --performance is required.",
            err=True,
        )
        raise SystemExit(1)

    run_tui(profiler_db=profiler, perf_data=performance)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
