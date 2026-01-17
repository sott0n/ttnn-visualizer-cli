"""Performance commands - display performance information."""

import click

from ..data.perf_csv import PerfCSV
from ..output.formatter import OutputFormat, OutputFormatter, format_ns


@click.group(invoke_without_command=True)
@click.option(
    "--performance",
    required=True,
    type=click.Path(exists=True),
    help="Path to performance report directory or CSV file",
)
@click.option(
    "--top",
    "-n",
    type=int,
    default=None,
    help="Show top N operations by execution time",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=100,
    help="Maximum number of operations to display (default: 100)",
)
@click.pass_context
def perf(
    ctx: click.Context,
    performance: str,
    top: int | None,
    limit: int,
) -> None:
    """Display performance information.

    Shows performance data from ops_perf_results CSV files including
    execution times, math utilization, and memory bandwidth.

    Use 'perf report' subcommand for detailed analysis.
    """
    # If a subcommand is invoked, let it handle the request
    if ctx.invoked_subcommand is not None:
        ctx.obj["performance_path"] = performance
        return

    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        perf_data = PerfCSV(performance)
        if not perf_data.is_valid():
            raise click.ClickException("No valid performance CSV found")
    except Exception as e:
        raise click.ClickException(str(e))

    if top:
        ops = perf_data.get_top_operations(top)
        title = f"Top {top} Operations by Execution Time"
    else:
        ops = perf_data.get_operations(limit=limit)
        title = "Performance Data"

    if not ops:
        click.echo("No performance data found")
        return

    if format_type == OutputFormat.JSON:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data, title=title))
    elif format_type == OutputFormat.CSV:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data))
    else:
        # Table format
        headers = ["Op Name", "Op Code", "Exec Time", "Cores", "Math Util %", "DRAM R BW %"]
        rows = []
        for op in ops:
            # Truncate op_name if too long
            op_name = op.op_name
            if len(op_name) > 40:
                op_name = op_name[:37] + "..."
            rows.append({
                "Op Name": op_name,
                "Op Code": op.op_code,
                "Exec Time": format_ns(op.execution_time_ns),
                "Cores": op.core_count,
                "Math Util %": f"{op.math_utilization:.1f}",
                "DRAM R BW %": f"{op.dram_read_bw:.1f}",
            })
        click.echo(formatter.format_output(rows, headers=headers, title=title))


@perf.command()
@click.option(
    "--signpost",
    "-s",
    type=str,
    default=None,
    help="Filter by signpost name",
)
@click.pass_context
def report(ctx: click.Context, signpost: str | None) -> None:
    """Display detailed performance report.

    Shows comprehensive performance analysis including summary
    statistics and breakdown by operation type.
    """
    performance_path = ctx.obj.get("performance_path")
    if not performance_path:
        raise click.UsageError("--performance option is required")

    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        perf_data = PerfCSV(performance_path)
        if not perf_data.is_valid():
            raise click.ClickException("No valid performance CSV found")
    except Exception as e:
        raise click.ClickException(str(e))

    summary = perf_data.get_summary()
    top_ops = perf_data.get_top_operations(10)

    if format_type == OutputFormat.JSON:
        data = {
            "summary": summary,
            "top_operations": [op.to_dict() for op in top_ops],
        }
        click.echo(formatter.format_output(data, title="Performance Report"))
    else:
        # Table format
        output_lines = []
        output_lines.append("Performance Report")
        output_lines.append("=" * 18)
        output_lines.append("")
        output_lines.append("Summary:")
        output_lines.append(f"  CSV File:        {summary.get('csv_file', 'N/A')}")
        output_lines.append(f"  Total Operations: {summary['total_operations']:,}")
        output_lines.append(f"  Total Exec Time:  {format_ns(summary['total_execution_time_ns'])}")
        output_lines.append(f"  Avg Exec Time:    {format_ns(summary['avg_execution_time_ns'])}")
        output_lines.append(f"  Max Exec Time:    {format_ns(summary['max_execution_time_ns'])}")
        output_lines.append(f"  Min Exec Time:    {format_ns(summary['min_execution_time_ns'])}")
        output_lines.append(f"  Avg Math Util:    {summary.get('avg_math_utilization', 0):.1f}%")

        if top_ops:
            output_lines.append("")
            output_lines.append("Top 10 Operations by Execution Time:")
            output_lines.append("-" * 40)
            for i, op in enumerate(top_ops, 1):
                op_name = op.op_name
                if len(op_name) > 35:
                    op_name = op_name[:32] + "..."
                output_lines.append(
                    f"  {i:2}. {op_name:<35} {format_ns(op.execution_time_ns):>12}"
                )

        click.echo("\n".join(output_lines))


@perf.command()
@click.pass_context
def summary(ctx: click.Context) -> None:
    """Display performance summary statistics."""
    performance_path = ctx.obj.get("performance_path")
    if not performance_path:
        raise click.UsageError("--performance option is required")

    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        perf_data = PerfCSV(performance_path)
        if not perf_data.is_valid():
            raise click.ClickException("No valid performance CSV found")
    except Exception as e:
        raise click.ClickException(str(e))

    summary_data = perf_data.get_summary()

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(summary_data, title="Performance Summary"))
    else:
        output_lines = []
        output_lines.append("Performance Summary")
        output_lines.append("=" * 19)
        output_lines.append("")
        output_lines.append(f"CSV File:           {summary_data.get('csv_file', 'N/A')}")
        output_lines.append(f"Total Operations:   {summary_data['total_operations']:,}")
        output_lines.append(f"Total Exec Time:    {format_ns(summary_data['total_execution_time_ns'])}")
        output_lines.append(f"Avg Exec Time:      {format_ns(summary_data['avg_execution_time_ns'])}")
        output_lines.append(f"Max Exec Time:      {format_ns(summary_data['max_execution_time_ns'])}")
        output_lines.append(f"Min Exec Time:      {format_ns(summary_data['min_execution_time_ns'])}")
        output_lines.append(f"Avg Math Util:      {summary_data.get('avg_math_utilization', 0):.1f}%")

        click.echo("\n".join(output_lines))
