"""Performance analysis subcommands for TTNN CLI."""

import click

from ..data.perf_analysis import PerfAnalyzer
from ..data.perf_csv import PerfCSV
from ..output.formatter import OutputFormat, OutputFormatter, format_ns


def _get_perf_data(ctx: click.Context) -> PerfCSV:
    """Get performance data from context."""
    performance_path = ctx.obj.get("performance_path")
    if not performance_path:
        raise click.UsageError("--performance option is required")

    try:
        perf_data = PerfCSV(performance_path)
        if not perf_data.is_valid():
            raise click.ClickException("No valid performance CSV found")
        return perf_data
    except Exception as e:
        raise click.ClickException(str(e))


def _get_format(ctx: click.Context) -> OutputFormat:
    """Get output format from context."""
    return OutputFormat(ctx.obj.get("format", "table"))


@click.group()
@click.pass_context
def analysis(ctx: click.Context) -> None:
    """Performance analysis commands.

    Analyze operation performance including distribution, efficiency,
    and bottleneck identification.
    """
    pass


@analysis.command(name="op-distribution")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    help="Maximum number of operation types to display",
)
@click.pass_context
def op_distribution(ctx: click.Context, limit: int) -> None:
    """Show operation type distribution.

    Displays distribution of operations by type, including count,
    total time, and percentage of overall execution time.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    distribution = analyzer.get_op_distribution(limit=limit)

    if not distribution:
        click.echo("No operation data found")
        return

    total_ops = sum(d.count for d in distribution)
    total_time = sum(d.total_time_ns for d in distribution)

    if format_type == OutputFormat.JSON:
        data = {
            "distribution": [d.to_dict() for d in distribution],
            "summary": {
                "total_operations": total_ops,
                "total_time_ns": total_time,
            },
        }
        click.echo(formatter.format_output(data, title="Operation Type Distribution"))
        return
    elif format_type == OutputFormat.CSV:
        data = [d.to_dict() for d in distribution]
        click.echo(formatter.format_output(data))
        return

    # Table format
    output_lines = []
    output_lines.append("Operation Type Distribution")
    output_lines.append("=" * 27)
    output_lines.append("")

    # Header
    output_lines.append(
        f"{'Op Code':<25} {'Count':>7} {'Total Time':>12} {'Avg Time':>12} {'% Time':>8} {'% Count':>8}"
    )
    output_lines.append("-" * 80)

    for d in distribution:
        output_lines.append(
            f"{d.op_code[:25]:<25} {d.count:>7} {format_ns(d.total_time_ns):>12} "
            f"{format_ns(d.avg_time_ns):>12} {d.percent_time:>7.1f}% {d.percent_count:>7.1f}%"
        )

    output_lines.append("")
    output_lines.append("Summary:")
    output_lines.append(f"  Total: {total_ops} operations, {format_ns(total_time)}")

    # Top 3 by time
    top3 = distribution[:3]
    if top3:
        top3_str = ", ".join(
            f"{d.op_code} ({d.percent_time:.1f}%)" for d in top3
        )
        output_lines.append(f"  Top 3 by time: {top3_str}")

    click.echo("\n".join(output_lines))


@analysis.command(name="core-efficiency")
@click.pass_context
def core_efficiency(ctx: click.Context) -> None:
    """Show core count efficiency analysis.

    Analyzes operations grouped by core count, showing utilization
    and bound distribution for each core configuration.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    efficiency_data = analyzer.get_core_efficiency()

    if not efficiency_data:
        click.echo("No core efficiency data found")
        return

    if format_type == OutputFormat.JSON:
        data = [e.to_dict() for e in efficiency_data]
        click.echo(formatter.format_output(data, title="Core Efficiency Analysis"))
        return
    elif format_type == OutputFormat.CSV:
        data = [e.to_dict() for e in efficiency_data]
        click.echo(formatter.format_output(data))
        return

    # Table format
    output_lines = []
    output_lines.append("Core Efficiency Analysis")
    output_lines.append("=" * 24)
    output_lines.append("")

    # Header
    output_lines.append(
        f"{'Cores':>6} {'Op Count':>9} {'Total Time':>12} {'Avg Time':>12} "
        f"{'Avg FPU%':>9} {'Bound Distribution':<30}"
    )
    output_lines.append("-" * 90)

    for e in efficiency_data:
        bound_dist = f"Compute: {e.compute_bound}, Memory: {e.memory_bound}, Balanced: {e.balanced}"
        output_lines.append(
            f"{e.core_count:>6} {e.op_count:>9} {format_ns(e.total_time_ns):>12} "
            f"{format_ns(e.avg_time_ns):>12} {e.avg_fpu_util:>8.1f}% {bound_dist:<30}"
        )

    # Generate insights
    output_lines.append("")
    output_lines.append("Insights:")

    # Find highest FPU utilization
    if efficiency_data:
        best_fpu = max(efficiency_data, key=lambda x: x.avg_fpu_util)
        if best_fpu.avg_fpu_util > 0:
            output_lines.append(
                f"  - {best_fpu.core_count}-core operations show highest FPU utilization ({best_fpu.avg_fpu_util:.1f}%)"
            )

        # Find potentially under-utilized configs
        low_util = [e for e in efficiency_data if 0 < e.avg_fpu_util < 30]
        for e in low_util[:2]:
            output_lines.append(
                f"  - {e.core_count}-core operations may be under-utilizing resources ({e.avg_fpu_util:.1f}% FPU)"
            )

        # Compute bound percentage
        total_ops = sum(e.op_count for e in efficiency_data)
        total_compute = sum(e.compute_bound for e in efficiency_data)
        if total_ops > 0 and total_compute > 0:
            output_lines.append(
                f"  - Most operations are compute-bound ({total_compute}/{total_ops} = {total_compute/total_ops*100:.0f}%)"
            )

    click.echo("\n".join(output_lines))


@analysis.command()
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    help="Maximum number of operations to display",
)
@click.pass_context
def matmul(ctx: click.Context, limit: int) -> None:
    """Analyze Matmul operations.

    Shows detailed analysis of Matmul operations including efficiency,
    utilization, and math fidelity breakdown.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    analysis_data = analyzer.get_matmul_analysis(limit=limit)

    if not analysis_data["operations"]:
        click.echo("No Matmul operations found")
        return

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(analysis_data, title="Matmul Operations Analysis"))
        return
    elif format_type == OutputFormat.CSV:
        click.echo(formatter.format_output(analysis_data["operations"]))
        return

    _print_op_analysis(analysis_data, "Matmul")


@analysis.command()
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    help="Maximum number of operations to display",
)
@click.pass_context
def conv(ctx: click.Context, limit: int) -> None:
    """Analyze Conv operations.

    Shows detailed analysis of convolution operations including efficiency,
    utilization, and math fidelity breakdown.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    analysis_data = analyzer.get_conv_analysis(limit=limit)

    if not analysis_data["operations"]:
        click.echo("No Conv operations found")
        return

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(analysis_data, title="Conv Operations Analysis"))
        return
    elif format_type == OutputFormat.CSV:
        click.echo(formatter.format_output(analysis_data["operations"]))
        return

    _print_op_analysis(analysis_data, "Conv")


def _print_op_analysis(analysis_data: dict, op_type: str) -> None:
    """Print operation analysis in table format."""
    output_lines = []
    output_lines.append(f"{op_type} Operations Analysis")
    output_lines.append("=" * (len(op_type) + 21))
    output_lines.append("")

    # Header
    output_lines.append(
        f"{'ID':>8} {'Cores':>6} {'Device Time':>12} {'Ideal Time':>12} "
        f"{'Efficiency':>11} {'FPU%':>7} {'Bound':<10}"
    )
    output_lines.append("-" * 75)

    for op in analysis_data["operations"]:
        op_id = op["global_call_count"] if op["global_call_count"] else "-"
        ideal = format_ns(op["ideal_time_ns"]) if op["ideal_time_ns"] else "-"
        eff = f"{op['efficiency']:.1f}%" if op["efficiency"] else "-"
        fpu = f"{op['fpu_util']:.1f}" if op["fpu_util"] > 0 else "-"
        bound = op["bound"] if op["bound"] else "-"

        output_lines.append(
            f"{str(op_id):>8} {op['core_count']:>6} {format_ns(op['device_time_ns']):>12} "
            f"{ideal:>12} {eff:>11} {fpu:>7} {bound:<10}"
        )

    # Summary
    summary = analysis_data["summary"]
    output_lines.append("")
    output_lines.append("Summary:")
    output_lines.append(f"  Total: {summary['total_count']} {op_type} operations")
    output_lines.append(
        f"  Total Time: {format_ns(summary['total_time_ns'])} ({summary['percent_of_all_ops']:.1f}% of all ops)"
    )
    output_lines.append(f"  Avg Efficiency: {summary['avg_efficiency']:.1f}% (Ideal/Device ratio)")
    output_lines.append(f"  Avg FPU Utilization: {summary['avg_fpu_util']:.1f}%")

    # Efficiency distribution
    eff_dist = analysis_data["efficiency_distribution"]
    total_eff = eff_dist["high"] + eff_dist["medium"] + eff_dist["low"]
    output_lines.append("")
    output_lines.append("Efficiency Distribution:")
    if total_eff > 0:
        output_lines.append(
            f"  High (>80%):    {eff_dist['high']:>3} ops ({eff_dist['high']/total_eff*100:.1f}%)"
        )
        output_lines.append(
            f"  Medium (50-80%): {eff_dist['medium']:>3} ops ({eff_dist['medium']/total_eff*100:.1f}%)"
        )
        output_lines.append(
            f"  Low (<50%):     {eff_dist['low']:>3} ops ({eff_dist['low']/total_eff*100:.1f}%)"
        )
    else:
        output_lines.append("  No efficiency data available")

    # Math fidelity
    fidelity = analysis_data["math_fidelity"]
    if fidelity:
        output_lines.append("")
        output_lines.append("Math Fidelity:")
        for f, count in sorted(fidelity.items(), key=lambda x: -x[1]):
            output_lines.append(f"  {f}: {count} ops")

    click.echo("\n".join(output_lines))


@analysis.command()
@click.option(
    "--efficiency-threshold",
    "-e",
    type=float,
    default=50.0,
    help="FPU utilization threshold for low efficiency (default: 50%)",
)
@click.option(
    "--gap-threshold",
    "-g",
    type=float,
    default=100.0,
    help="Op-to-op gap threshold in milliseconds (default: 100ms)",
)
@click.pass_context
def bottlenecks(
    ctx: click.Context, efficiency_threshold: float, gap_threshold: float
) -> None:
    """Identify performance bottlenecks.

    Finds operations with low efficiency, high op-to-op gaps,
    and memory-bound operations with low DRAM utilization.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    bottleneck_data = analyzer.get_bottlenecks(
        efficiency_threshold=efficiency_threshold,
        gap_threshold_ms=gap_threshold,
    )

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(bottleneck_data, title="Performance Bottlenecks"))
        return
    elif format_type == OutputFormat.CSV:
        # Flatten all bottlenecks for CSV
        all_bottlenecks = (
            bottleneck_data["low_efficiency"]
            + bottleneck_data["high_gap"]
            + bottleneck_data["memory_inefficient"]
        )
        click.echo(formatter.format_output(all_bottlenecks))
        return

    # Table format
    output_lines = []
    output_lines.append("Performance Bottlenecks")
    output_lines.append("=" * 23)

    # Low efficiency operations
    low_eff = bottleneck_data["low_efficiency"]
    if low_eff:
        output_lines.append("")
        output_lines.append(f"Low Efficiency Operations (<{efficiency_threshold:.0f}%):")
        output_lines.append(
            f"  {'ID':>8}  {'Op Code':<20}  {'Device Time':>12}  {'Efficiency':>11}  {'Issue':<30}"
        )
        output_lines.append("  " + "-" * 90)
        for b in low_eff[:10]:
            op_id = b["global_call_count"] if b["global_call_count"] else "-"
            eff = f"{b['efficiency']:.1f}%" if b["efficiency"] else "-"
            output_lines.append(
                f"  {str(op_id):>8}  {b['op_code'][:20]:<20}  "
                f"{format_ns(b['device_time_ns']):>12}  {eff:>11}  {b['issue'][:30]:<30}"
            )

    # High gap operations
    high_gap = bottleneck_data["high_gap"]
    if high_gap:
        output_lines.append("")
        output_lines.append(f"High Op-to-Op Gap (>{gap_threshold:.0f}ms):")
        output_lines.append(
            f"  {'ID':>8}  {'Op Code':<20}  {'Gap':>12}  {'Possible Cause':<30}"
        )
        output_lines.append("  " + "-" * 80)
        for b in high_gap[:10]:
            op_id = b["global_call_count"] if b["global_call_count"] else "-"
            output_lines.append(
                f"  {str(op_id):>8}  {b['op_code'][:20]:<20}  "
                f"{format_ns(b['device_time_ns']):>12}  {b['issue'][:30]:<30}"
            )

    # Memory inefficient operations
    mem_ineff = bottleneck_data["memory_inefficient"]
    if mem_ineff:
        output_lines.append("")
        output_lines.append("Memory-Bound Operations with Low DRAM Utilization:")
        output_lines.append(
            f"  {'ID':>8}  {'Op Code':<20}  {'Device Time':>12}  {'DRAM%':>8}  {'Issue':<30}"
        )
        output_lines.append("  " + "-" * 90)
        for b in mem_ineff[:10]:
            op_id = b["global_call_count"] if b["global_call_count"] else "-"
            dram = f"{b['efficiency']:.1f}%" if b["efficiency"] else "-"
            output_lines.append(
                f"  {str(op_id):>8}  {b['op_code'][:20]:<20}  "
                f"{format_ns(b['device_time_ns']):>12}  {dram:>8}  {b['issue'][:30]:<30}"
            )

    # Summary
    summary = bottleneck_data["summary"]
    output_lines.append("")
    output_lines.append("Summary:")
    output_lines.append(f"  Low efficiency operations: {summary['low_efficiency_count']}")
    output_lines.append(f"  High op-to-op gap operations: {summary['high_gap_count']}")
    output_lines.append(
        f"  Memory-bound low utilization: {summary['memory_inefficient_count']}"
    )

    if not (low_eff or high_gap or mem_ineff):
        output_lines.append("")
        output_lines.append("No significant bottlenecks detected.")

    click.echo("\n".join(output_lines))


@analysis.command(name="summary")
@click.pass_context
def analysis_summary(ctx: click.Context) -> None:
    """Show overall performance analysis summary.

    Provides a comprehensive overview of performance metrics including
    operation distribution, utilization, and potential issues.
    """
    perf_data = _get_perf_data(ctx)
    format_type = _get_format(ctx)
    formatter = OutputFormatter(format_type)

    ops = perf_data.get_operations(limit=None)
    analyzer = PerfAnalyzer(ops)
    summary = analyzer.get_summary()

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(summary.to_dict(), title="Performance Analysis Summary"))
        return
    elif format_type == OutputFormat.CSV:
        click.echo(formatter.format_output(summary.to_dict()))
        return

    # Table format
    output_lines = []
    output_lines.append("Performance Analysis Summary")
    output_lines.append("=" * 28)
    output_lines.append("")

    # Overview
    output_lines.append("Overview:")
    output_lines.append(f"  Total Operations: {summary.total_operations}")
    output_lines.append(f"  Total Device Time: {format_ns(summary.total_device_time_ns)}")
    output_lines.append(f"  Total Op-to-Op Gap: {format_ns(summary.total_op_to_op_gap_ns)}")

    # Operation distribution
    total_bound = (
        summary.compute_bound_count
        + summary.memory_bound_count
        + summary.balanced_count
    )
    output_lines.append("")
    output_lines.append("Operation Distribution:")
    if total_bound > 0:
        output_lines.append(
            f"  Compute-bound: {summary.compute_bound_count} ops "
            f"({summary.compute_bound_count/total_bound*100:.1f}%)"
        )
        output_lines.append(
            f"  Memory-bound: {summary.memory_bound_count} ops "
            f"({summary.memory_bound_count/total_bound*100:.1f}%)"
        )
        output_lines.append(
            f"  Balanced: {summary.balanced_count} ops "
            f"({summary.balanced_count/total_bound*100:.1f}%)"
        )
    else:
        output_lines.append("  No bound data available")

    # Top op codes
    output_lines.append("")
    output_lines.append("Top Op Codes by Time:")
    for i, (op_code, count, time_ns, percent) in enumerate(summary.top_op_codes, 1):
        output_lines.append(
            f"  {i}. {op_code:<15} ({count} ops): {format_ns(time_ns):>12} ({percent:.1f}%)"
        )

    # Utilization
    output_lines.append("")
    output_lines.append("Utilization:")
    output_lines.append(f"  Avg FPU Utilization: {summary.avg_fpu_util:.1f}%")
    output_lines.append(f"  Avg DRAM Utilization: {summary.avg_dram_util:.1f}%")

    # Potential issues
    output_lines.append("")
    output_lines.append("Potential Issues:")
    if summary.low_efficiency_count > 0:
        output_lines.append(
            f"  - {summary.low_efficiency_count} operations with <50% efficiency"
        )
    if summary.high_gap_count > 0:
        output_lines.append(
            f"  - {summary.high_gap_count} operations with op-to-op gap >100ms"
        )
    if summary.low_efficiency_count == 0 and summary.high_gap_count == 0:
        output_lines.append("  No significant issues detected")

    click.echo("\n".join(output_lines))
