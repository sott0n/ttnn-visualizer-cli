"""Performance commands - display performance information."""

import click
from tabulate import tabulate

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


@perf.command(name="perf-report")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=100,
    help="Maximum number of operations to display (default: 100)",
)
@click.option(
    "--sort-by",
    "-s",
    type=click.Choice([
        "id", "device_time", "op_to_op_gap", "cores",
        "dram_percent", "flops_percent", "op_code"
    ]),
    default=None,
    help="Sort results by column",
)
@click.option(
    "--desc/--asc",
    default=True,
    help="Sort direction (default: descending)",
)
@click.option(
    "--op-code",
    "-o",
    type=str,
    default=None,
    help="Filter by op code (case-insensitive, partial match)",
)
@click.option(
    "--device",
    "-d",
    type=int,
    default=None,
    help="Filter by device ID",
)
@click.option(
    "--buffer-type",
    "-b",
    type=click.Choice(["L1", "DRAM", "System"], case_sensitive=False),
    default=None,
    help="Filter by buffer type",
)
@click.option(
    "--bound",
    "-B",
    type=click.Choice(["Compute", "Memory", "Balanced"], case_sensitive=False),
    default=None,
    help="Filter by bound type",
)
@click.option(
    "--min-time",
    type=float,
    default=None,
    help="Minimum device time in microseconds",
)
@click.option(
    "--max-time",
    type=float,
    default=None,
    help="Maximum device time in microseconds",
)
@click.pass_context
def perf_report(
    ctx: click.Context,
    limit: int,
    sort_by: str | None,
    desc: bool,
    op_code: str | None,
    device: int | None,
    buffer_type: str | None,
    bound: str | None,
    min_time: float | None,
    max_time: float | None,
) -> None:
    """Display detailed performance report table.

    Shows comprehensive performance data with columns:
    ID, Total, Bound, Op Code, Device ID, Buffer Type, Layout,
    Device Time, Op-to-Op Gap, Cores, DRAM, DRAM %, FLOPs, FLOPs %, Math Fidelity

    Includes summary totals at the bottom.

    Examples:
        # Sort by device time descending
        perf --performance ./perf perf-report --sort-by device_time

        # Filter by op code
        perf --performance ./perf perf-report --op-code Matmul

        # Filter compute-bound operations with min time
        perf --performance ./perf perf-report --bound Compute --min-time 10
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

    # Get all operations first (we'll apply limit after filtering/sorting)
    ops = perf_data.get_operations(limit=None)
    # Filter out signpost rows
    ops = [op for op in ops if op.op_name != "signpost"]

    # Apply filters
    if op_code:
        ops = [op for op in ops if op_code.lower() in (op.op_code or "").lower()]

    if device is not None:
        ops = [op for op in ops if op.device_id == device]

    if buffer_type:
        ops = [op for op in ops if (op.buffer_type or "").upper() == buffer_type.upper()]

    if bound:
        ops = [op for op in ops if (op.bound or "").lower() == bound.lower()]

    if min_time is not None:
        # Convert microseconds to nanoseconds for comparison
        min_time_ns = min_time * 1000
        ops = [op for op in ops if op.execution_time_ns >= min_time_ns]

    if max_time is not None:
        # Convert microseconds to nanoseconds for comparison
        max_time_ns = max_time * 1000
        ops = [op for op in ops if op.execution_time_ns <= max_time_ns]

    # Apply sorting
    if sort_by:
        sort_key_map = {
            "id": lambda x: x.global_call_count or 0,
            "device_time": lambda x: x.execution_time_ns,
            "op_to_op_gap": lambda x: x.op_to_op_gap_ns,
            "cores": lambda x: x.core_count or 0,
            "dram_percent": lambda x: x.dram_bw_util_percent or 0,
            "flops_percent": lambda x: x.fpu_util_percent or 0,
            "op_code": lambda x: (x.op_code or "").lower(),
        }
        if sort_by in sort_key_map:
            ops.sort(key=sort_key_map[sort_by], reverse=desc)

    # Apply limit after filtering and sorting
    if limit:
        ops = ops[:limit]

    if not ops:
        click.echo("No performance data found")
        return

    if format_type == OutputFormat.JSON:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data, title="Performance Report"))
        return
    elif format_type == OutputFormat.CSV:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data))
        return

    # Table format with detailed columns
    headers = [
        "ID",
        "Total",
        "Bound",
        "Op Code",
        "Device ID",
        "Buffer Type",
        "Layout",
        "Device Time",
        "Op-to-Op Gap",
        "Cores",
        "DRAM",
        "DRAM %",
        "FLOPs",
        "FLOPs %",
        "Math Fidelity",
    ]

    rows = []
    total_device_time = 0.0
    total_op_to_op_gap = 0.0

    for i, op in enumerate(ops, 1):
        total_device_time += op.execution_time_ns
        total_op_to_op_gap += op.op_to_op_gap_ns

        # Format DRAM bandwidth
        dram_bw = op.dram_bandwidth
        dram_str = f"{dram_bw:.1f}" if dram_bw is not None else "-"

        # Format FLOPs
        flops = op.flops
        flops_str = f"{flops:.0f}" if flops is not None else "-"

        rows.append([
            op.global_call_count if op.global_call_count else i,
            i,  # Total (row number)
            op.bound if op.bound else "-",
            op.op_code if op.op_code else "-",
            op.device_id,
            op.buffer_type if op.buffer_type else "-",
            op.layout if op.layout else "-",
            format_ns(op.execution_time_ns),
            format_ns(op.op_to_op_gap_ns) if op.op_to_op_gap_ns > 0 else "-",
            op.core_count if op.core_count else "-",
            dram_str,
            f"{op.dram_bw_util_percent:.1f}" if op.dram_bw_util_percent else "-",
            flops_str,
            f"{op.fpu_util_percent:.1f}" if op.fpu_util_percent else "-",
            op.math_fidelity if op.math_fidelity else "-",
        ])

    # Add summary row with proper column alignment
    summary_row = [
        "Total",                                      # ID
        len(ops),                                     # Total
        "",                                           # Bound
        f"{len(set(op.op_code for op in ops))} types",  # Op Code
        "",                                           # Device ID
        "",                                           # Buffer Type
        "",                                           # Layout
        format_ns(total_device_time),                 # Device Time
        format_ns(total_op_to_op_gap),                # Op-to-Op Gap
        "",                                           # Cores
        "",                                           # DRAM
        "",                                           # DRAM %
        "",                                           # FLOPs
        "",                                           # FLOPs %
        "",                                           # Math Fidelity
    ]

    click.echo("Performance Report")
    click.echo("=" * 18)
    click.echo()
    # Add rows with summary
    all_rows = rows + [["─" * 8] * len(headers), summary_row]
    click.echo(tabulate(all_rows, headers=headers, tablefmt="simple"))
