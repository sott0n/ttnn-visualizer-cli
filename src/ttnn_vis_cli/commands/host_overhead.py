"""Host overhead analysis commands for TTNN Visualizer CLI."""

import csv
import io
from pathlib import Path

import click

from ..data.host_overhead_analysis import HostOverheadAnalyzer
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
def host_overhead(ctx: click.Context, performance: Path) -> None:
    """Host overhead analysis commands.

    Analyze host vs device time to determine Metal Trace applicability.
    """
    ctx.ensure_object(dict)
    ctx.obj["performance"] = performance


@host_overhead.command(name="summary")
@click.pass_context
def overhead_summary(ctx: click.Context) -> None:
    """Show host overhead summary with Metal Trace recommendations.

    Example:
        ttnn-vis-cli host-overhead -P samples/performance summary
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = HostOverheadAnalyzer(operations)
    summary = analyzer.get_summary()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(summary.to_dict(), indent=2))
    else:
        click.echo("=== Host Overhead Analysis ===\n")

        # Time breakdown
        click.echo("Time Breakdown:")
        click.echo(f"  Device Time:     {summary.total_device_time_ns/1_000_000:.3f} ms")
        click.echo(f"  Op-to-Op Gap:    {summary.total_op_to_op_gap_ns/1_000_000:.3f} ms")
        click.echo(f"  Total E2E Time:  {summary.total_e2e_time_ns/1_000_000:.3f} ms")
        click.echo("")

        # Utilization
        click.echo("Utilization:")
        click.echo(f"  Device Utilization:  {summary.device_utilization_percent:.1f}%")
        click.echo(f"  Host Overhead:       {summary.host_overhead_percent:.1f}%")
        click.echo("")

        # Statistics
        click.echo("Statistics:")
        click.echo(f"  Operations:       {summary.operation_count}")
        click.echo(f"  Avg Op-to-Op Gap: {summary.avg_op_to_op_gap_ns/1_000:.1f} us")
        click.echo(f"  Max Op-to-Op Gap: {summary.max_op_to_op_gap_ns/1_000:.1f} us")
        click.echo("")

        # Status
        click.echo("Status:")
        bound_status = "HOST-BOUND" if summary.is_host_bound else "DEVICE-BOUND"
        click.echo(f"  Model Status:        {bound_status}")
        trace_status = "YES" if summary.metal_trace_recommended else "NO"
        click.echo(f"  Metal Trace Recommended: {trace_status}")
        click.echo("")

        # Recommendations
        click.echo("Recommendations:")
        for rec in summary.recommendations:
            click.echo(f"  - {rec}")


@host_overhead.command(name="top")
@click.option("--limit", "-n", type=int, default=20, help="Number of operations to show")
@click.pass_context
def overhead_top(ctx: click.Context, limit: int) -> None:
    """Show operations with highest host overhead.

    Example:
        ttnn-vis-cli host-overhead -P samples/performance top --limit 10
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = HostOverheadAnalyzer(operations)
    top_ops = analyzer.get_top_overhead_operations(limit=limit)

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([op.to_dict() for op in top_ops], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "op_code", "op_name", "device_time_us", "op_to_op_gap_us",
            "overhead_percent", "core_count"
        ])
        for op in top_ops:
            writer.writerow([
                op.op_code,
                op.op_name,
                f"{op.device_time_ns/1000:.2f}",
                f"{op.op_to_op_gap_ns/1000:.2f}",
                f"{op.overhead_percent:.1f}",
                op.core_count,
            ])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not top_ops:
            click.echo("No operations found.")
            return

        headers = ["Op Code", "Op Name", "Device (us)", "Gap (us)", "Overhead %", "Cores"]
        rows = []
        for op in top_ops:
            rows.append([
                op.op_code[:20] + "..." if len(op.op_code) > 20 else op.op_code,
                op.op_name[:25] + "..." if len(op.op_name) > 25 else op.op_name,
                f"{op.device_time_ns/1000:.1f}",
                f"{op.op_to_op_gap_ns/1000:.1f}",
                f"{op.overhead_percent:.1f}%",
                op.core_count,
            ])

        click.echo(f"Top {len(top_ops)} operations by op-to-op gap:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))


@host_overhead.command(name="distribution")
@click.pass_context
def overhead_distribution(ctx: click.Context) -> None:
    """Show distribution of operations by overhead level.

    Example:
        ttnn-vis-cli host-overhead -P samples/performance distribution
    """
    perf_path = ctx.obj["performance"]
    output_format = get_format(ctx)

    perf_csv = PerfCSV(perf_path)
    if not perf_csv.is_valid():
        click.echo("Error: No valid performance CSV found.", err=True)
        raise SystemExit(1)

    operations = perf_csv.get_operations()
    analyzer = HostOverheadAnalyzer(operations)
    distribution = analyzer.get_overhead_distribution()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(distribution, indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["overhead_range", "count"])
        for range_name, count in distribution.items():
            writer.writerow([range_name, count])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        if not distribution:
            click.echo("No operations found.")
            return

        total = sum(distribution.values())
        headers = ["Overhead Range", "Count", "Percent"]
        rows = []
        for range_name, count in distribution.items():
            percent = (count / total * 100) if total > 0 else 0
            rows.append([range_name, count, f"{percent:.1f}%"])

        click.echo("Operations by overhead level:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
        click.echo(f"\nTotal operations: {total}")
