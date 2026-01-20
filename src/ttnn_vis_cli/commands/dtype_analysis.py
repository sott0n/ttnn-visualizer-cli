"""Data format analysis commands for TTNN Visualizer CLI."""

import csv
import io
from pathlib import Path
from typing import Optional

import click

from ..data.dtype_analysis import DataFormatAnalyzer, MathFidelityAnalyzer
from ..data.perf_csv import PerfCSV
from ..data.profiler_db import ProfilerDB
from ..output.formatter import OutputFormat


def get_format(ctx: click.Context) -> OutputFormat:
    """Get output format from context."""
    format_str = ctx.obj.get("format", "table") if ctx.obj else "table"
    return OutputFormat(format_str)


@click.group()
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
@click.pass_context
def dtype_analysis(
    ctx: click.Context,
    profiler: Optional[Path],
    performance: Optional[Path],
) -> None:
    """Data format analysis commands.

    Analyze tensor data types, layouts, and math fidelity settings.
    """
    ctx.ensure_object(dict)
    ctx.obj["profiler"] = profiler
    ctx.obj["performance"] = performance


@dtype_analysis.command(name="summary")
@click.pass_context
def dtype_summary(ctx: click.Context) -> None:
    """Show data format summary with optimization recommendations.

    Example:
        ttnn-vis-cli dtype-analysis -p samples/memory/db.sqlite summary
        ttnn-vis-cli dtype-analysis -p samples/memory/db.sqlite -P samples/performance summary
    """
    profiler_path = ctx.obj.get("profiler")
    perf_path = ctx.obj.get("performance")
    output_format = get_format(ctx)

    if not profiler_path:
        click.echo("Error: --profiler is required for summary.", err=True)
        raise SystemExit(1)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()
    analyzer = DataFormatAnalyzer(tensors)
    summary = analyzer.get_summary()

    if output_format == OutputFormat.JSON:
        import json

        result = summary.to_dict()

        # Add math fidelity if performance data available
        if perf_path:
            perf_csv = PerfCSV(perf_path)
            if perf_csv.is_valid():
                operations = perf_csv.get_operations()
                fidelity_analyzer = MathFidelityAnalyzer(operations)
                fidelity_summary = fidelity_analyzer.get_summary()
                result["math_fidelity"] = fidelity_summary.to_dict()

        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("=== Data Format Analysis ===\n")

        # DType distribution
        click.echo("Data Type Distribution:")
        for d in summary.dtype_distribution:
            click.echo(f"  {d.dtype}: {d.count} ({d.percent:.1f}%)")
        click.echo("")

        # Layout distribution
        click.echo("Layout Distribution:")
        for d in summary.layout_distribution:
            click.echo(f"  {d.layout}: {d.count} ({d.percent:.1f}%)")
        click.echo("")

        # Key metrics
        click.echo("Key Metrics:")
        click.echo(f"  Total Tensors:      {summary.total_tensors}")
        click.echo(f"  bfloat8_b Usage:    {summary.bfloat8_b_usage_percent:.1f}%")
        click.echo(f"  TILE Layout Usage:  {summary.tile_layout_percent:.1f}%")
        click.echo("")

        # Math fidelity if available
        if perf_path:
            perf_csv = PerfCSV(perf_path)
            if perf_csv.is_valid():
                operations = perf_csv.get_operations()
                fidelity_analyzer = MathFidelityAnalyzer(operations)
                fidelity_summary = fidelity_analyzer.get_summary()

                if fidelity_summary.fidelity_distribution:
                    click.echo("Math Fidelity Distribution:")
                    for d in fidelity_summary.fidelity_distribution:
                        click.echo(f"  {d.fidelity}: {d.count} ({d.percent:.1f}%)")
                    click.echo("")

        # Recommendations
        click.echo("Recommendations:")
        for rec in summary.recommendations:
            click.echo(f"  - {rec}")

        # Math fidelity recommendations
        if perf_path:
            perf_csv = PerfCSV(perf_path)
            if perf_csv.is_valid():
                operations = perf_csv.get_operations()
                fidelity_analyzer = MathFidelityAnalyzer(operations)
                fidelity_summary = fidelity_analyzer.get_summary()
                for rec in fidelity_summary.recommendations:
                    if "No math fidelity" not in rec:
                        click.echo(f"  - {rec}")


@dtype_analysis.command(name="dtypes")
@click.pass_context
def dtype_distribution(ctx: click.Context) -> None:
    """Show data type distribution.

    Example:
        ttnn-vis-cli dtype-analysis -p samples/memory/db.sqlite dtypes
    """
    profiler_path = ctx.obj.get("profiler")
    output_format = get_format(ctx)

    if not profiler_path:
        click.echo("Error: --profiler is required.", err=True)
        raise SystemExit(1)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()
    analyzer = DataFormatAnalyzer(tensors)
    distribution = analyzer.get_dtype_distribution()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([d.to_dict() for d in distribution], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["dtype", "count", "percent"])
        for d in distribution:
            writer.writerow([d.dtype, d.count, f"{d.percent:.1f}"])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not distribution:
            click.echo("No tensors found.")
            return

        headers = ["Data Type", "Count", "Percent"]
        rows = [[d.dtype, d.count, f"{d.percent:.1f}%"] for d in distribution]
        click.echo("Data Type Distribution:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))


@dtype_analysis.command(name="layouts")
@click.pass_context
def layout_distribution(ctx: click.Context) -> None:
    """Show layout distribution.

    Example:
        ttnn-vis-cli dtype-analysis -p samples/memory/db.sqlite layouts
    """
    profiler_path = ctx.obj.get("profiler")
    output_format = get_format(ctx)

    if not profiler_path:
        click.echo("Error: --profiler is required.", err=True)
        raise SystemExit(1)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()
    analyzer = DataFormatAnalyzer(tensors)
    distribution = analyzer.get_layout_distribution()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([d.to_dict() for d in distribution], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["layout", "count", "percent"])
        for d in distribution:
            writer.writerow([d.layout, d.count, f"{d.percent:.1f}"])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not distribution:
            click.echo("No tensors found.")
            return

        headers = ["Layout", "Count", "Percent"]
        rows = [[d.layout, d.count, f"{d.percent:.1f}%"] for d in distribution]
        click.echo("Layout Distribution:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))


@dtype_analysis.command(name="fidelity")
@click.pass_context
def math_fidelity(ctx: click.Context) -> None:
    """Show math fidelity distribution from performance data.

    Example:
        ttnn-vis-cli dtype-analysis -P samples/performance fidelity
    """
    perf_path = ctx.obj.get("performance")
    output_format = get_format(ctx)

    if not perf_path:
        click.echo("Error: --performance is required for fidelity analysis.", err=True)
        raise SystemExit(1)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = MathFidelityAnalyzer(operations)
    summary = analyzer.get_summary()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(summary.to_dict(), indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["fidelity", "count", "percent"])
        for d in summary.fidelity_distribution:
            writer.writerow([d.fidelity, d.count, f"{d.percent:.1f}"])
        click.echo(output.getvalue().rstrip())
    else:
        if not summary.fidelity_distribution:
            click.echo("No math fidelity data found in performance report.")
            return

        from tabulate import tabulate

        click.echo("Math Fidelity Distribution:\n")
        headers = ["Fidelity", "Count", "Percent"]
        rows = [
            [d.fidelity, d.count, f"{d.percent:.1f}%"]
            for d in summary.fidelity_distribution
        ]
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
        click.echo(f"\nTotal operations with fidelity data: {summary.total_operations}")
        click.echo("")
        click.echo("Recommendations:")
        for rec in summary.recommendations:
            click.echo(f"  - {rec}")
