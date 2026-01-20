"""Multi-CQ analysis commands for TTNN Visualizer CLI."""

import csv
import io
from pathlib import Path

import click

from ..data.multi_cq_analysis import MultiCQAnalyzer
from ..data.perf_csv import PerfCSV
from ..output.formatter import OutputFormat


def get_format(ctx: click.Context) -> OutputFormat:
    """Get output format from context."""
    format_str = ctx.obj.get("format", "table") if ctx.obj else "table"
    return OutputFormat(format_str)


@click.group()
@click.option(
    "--performance",
    "-P",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to performance report directory or CSV file",
)
@click.pass_context
def multi_cq(ctx: click.Context, performance: Path) -> None:
    """Multi-CQ analysis commands.

    Analyze command queue usage and I/O overlap efficiency.
    Helps determine if using 2 command queues (2CQ) would benefit performance.
    """
    ctx.ensure_object(dict)
    ctx.obj["performance"] = performance


@multi_cq.command(name="summary")
@click.pass_context
def cq_summary(ctx: click.Context) -> None:
    """Show multi-CQ analysis summary with 2CQ recommendations.

    Example:
        ttnn-vis-cli multi-cq -P samples/performance summary
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = MultiCQAnalyzer(operations)
    summary = analyzer.get_summary()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(summary.to_dict(), indent=2))
    else:
        click.echo("=== Multi-CQ Analysis ===\n")

        # I/O Timing Summary
        click.echo("I/O Timing Summary:")
        click.echo(f"  Total Device Time:     {summary.total_device_time_ns/1_000_000:.3f} ms")
        click.echo(f"  Total I/O Time:        {summary.total_io_time_ns/1_000_000:.3f} ms")
        click.echo(f"    - Dispatch CQ Time:  {summary.total_dispatch_cq_time_ns/1_000_000:.3f} ms")
        click.echo(f"    - Wait Time:         {summary.total_wait_time_ns/1_000_000:.3f} ms")
        click.echo(f"    - ERISC Time:        {summary.total_erisc_time_ns/1_000_000:.3f} ms")
        click.echo(f"  I/O Overhead:          {summary.io_overhead_percent:.1f}%")
        click.echo("")

        # Analysis
        click.echo("Analysis:")
        click.echo(f"  Total Operations:      {summary.total_operations}")
        io_pct = (
            summary.io_bound_operations / summary.total_operations * 100
            if summary.total_operations > 0
            else 0
        )
        compute_bound = summary.total_operations - summary.io_bound_operations
        compute_pct = 100 - io_pct
        click.echo(f"  I/O-bound Operations:  {summary.io_bound_operations} ({io_pct:.1f}%)")
        click.echo(f"  Compute-bound:         {compute_bound} ({compute_pct:.1f}%)")
        click.echo("")

        # Status
        click.echo("Status:")
        bound_status = "I/O-BOUND" if summary.is_io_bound else "COMPUTE-BOUND"
        click.echo(f"  Model Status:         {bound_status}")
        cq_status = "YES" if summary.multi_cq_recommended else "NO"
        click.echo(f"  2CQ Recommended:      {cq_status}")
        click.echo("")

        # Recommendations
        click.echo("Recommendations:")
        for rec in summary.recommendations:
            click.echo(f"  - {rec}")


@multi_cq.command(name="io-bound")
@click.option("--limit", "-n", type=int, default=20, help="Number of operations to show")
@click.pass_context
def io_bound_ops(ctx: click.Context, limit: int) -> None:
    """Show operations with highest I/O overhead.

    Example:
        ttnn-vis-cli multi-cq -P samples/performance io-bound --limit 10
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = MultiCQAnalyzer(operations)
    io_ops = analyzer.get_io_bound_operations(limit=limit)

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([op.to_dict() for op in io_ops], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "op_code", "op_name", "device_time_us", "io_time_us",
            "io_overhead_pct", "dispatch_us", "wait_us", "erisc_us"
        ])
        for op in io_ops:
            writer.writerow([
                op.op_code,
                op.op_name,
                f"{op.device_time_ns/1000:.2f}",
                f"{op.total_io_time_ns/1000:.2f}",
                f"{op.io_overhead_percent:.1f}",
                f"{op.dispatch_time_ns/1000:.2f}",
                f"{op.wait_time_ns/1000:.2f}",
                f"{op.erisc_time_ns/1000:.2f}",
            ])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not io_ops:
            click.echo("No operations found.")
            return

        headers = ["Op Code", "Device (us)", "I/O (us)", "I/O %", "Dispatch", "Wait", "ERISC"]
        rows = []
        for op in io_ops:
            rows.append([
                op.op_code[:20] + "..." if len(op.op_code) > 20 else op.op_code,
                f"{op.device_time_ns/1000:.1f}",
                f"{op.total_io_time_ns/1000:.1f}",
                f"{op.io_overhead_percent:.1f}%",
                f"{op.dispatch_time_ns/1000:.1f}",
                f"{op.wait_time_ns/1000:.1f}",
                f"{op.erisc_time_ns/1000:.1f}",
            ])

        click.echo(f"Top {len(io_ops)} I/O-Bound Operations:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))


@multi_cq.command(name="distribution")
@click.pass_context
def io_distribution(ctx: click.Context) -> None:
    """Show distribution of operations by I/O overhead level.

    Example:
        ttnn-vis-cli multi-cq -P samples/performance distribution
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = MultiCQAnalyzer(operations)
    distribution = analyzer.get_io_distribution()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(distribution, indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["io_overhead_range", "count"])
        for range_name, count in distribution.items():
            writer.writerow([range_name, count])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not distribution:
            click.echo("No operations found.")
            return

        total = sum(distribution.values())
        headers = ["I/O Overhead Range", "Count", "Percent"]
        rows = []
        for range_name, count in distribution.items():
            percent = (count / total * 100) if total > 0 else 0
            rows.append([range_name, count, f"{percent:.1f}%"])

        click.echo("I/O Overhead Distribution:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
        click.echo(f"\nTotal operations: {total}")
