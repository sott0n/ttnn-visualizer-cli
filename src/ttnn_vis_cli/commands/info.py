"""Info command - display report overview."""

import click

from ..data.profiler_db import ProfilerDB
from ..data.perf_csv import PerfCSV
from ..output.formatter import OutputFormat, OutputFormatter, format_bytes, format_ns


@click.command()
@click.option(
    "--profiler",
    "-p",
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.option(
    "--performance",
    type=click.Path(exists=True),
    help="Path to performance report directory or CSV file",
)
@click.pass_context
def info(ctx: click.Context, profiler: str | None, performance: str | None) -> None:
    """Display report overview and summary information.

    Shows basic statistics about the profiling report including
    operation count, tensor count, device information, and total duration.
    """
    if not profiler and not performance:
        raise click.UsageError("At least one of --profiler or --performance is required")

    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    result = {}

    if profiler:
        try:
            db = ProfilerDB(profiler)
            report_info = db.get_report_info()
            result["profiler"] = {
                "path": str(profiler),
                "operation_count": report_info.operation_count,
                "tensor_count": report_info.tensor_count,
                "buffer_count": report_info.buffer_count,
                "device_count": report_info.device_count,
                "total_duration_ns": report_info.total_duration_ns,
                "total_duration_ms": round(report_info.total_duration_ns / 1_000_000, 3),
                "devices": [d.to_dict() for d in report_info.devices],
            }
        except FileNotFoundError as e:
            raise click.ClickException(str(e))

    if performance:
        try:
            perf = PerfCSV(performance)
            if perf.is_valid():
                summary = perf.get_summary()
                result["performance"] = summary
            else:
                result["performance"] = {"error": "No valid performance CSV found"}
        except Exception as e:
            result["performance"] = {"error": str(e)}

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(result))
    else:
        # Table format - print sections
        output_lines = []

        if "profiler" in result:
            p = result["profiler"]
            output_lines.append("Profiler Report Summary")
            output_lines.append("=" * 24)
            output_lines.append(f"Path:           {p['path']}")
            output_lines.append(f"Operations:     {p['operation_count']:,}")
            output_lines.append(f"Tensors:        {p['tensor_count']:,}")
            output_lines.append(f"Buffers:        {p['buffer_count']:,}")
            output_lines.append(f"Devices:        {p['device_count']}")
            output_lines.append(f"Total Duration: {format_ns(p['total_duration_ns'])}")

            if p["devices"]:
                output_lines.append("")
                output_lines.append("Devices:")
                for dev in p["devices"]:
                    output_lines.append(f"  Device {dev['id']}:")
                    output_lines.append(f"    Compute Cores: {dev['total_compute_cores']} ({dev['num_x_compute_cores']}x{dev['num_y_compute_cores']})")
                    output_lines.append(f"    L1 Memory:     {format_bytes(dev['total_l1_memory'])}")
                    output_lines.append(f"    L1 for Tensors:{format_bytes(dev['total_l1_for_tensors'])}")

        if "performance" in result:
            if output_lines:
                output_lines.append("")
            perf_data = result["performance"]
            output_lines.append("Performance Report Summary")
            output_lines.append("=" * 26)
            if "error" in perf_data:
                output_lines.append(f"Error: {perf_data['error']}")
            else:
                output_lines.append(f"CSV File:        {perf_data.get('csv_file', 'N/A')}")
                output_lines.append(f"Operations:      {perf_data['total_operations']:,}")
                output_lines.append(f"Total Exec Time: {format_ns(perf_data['total_execution_time_ns'])}")
                output_lines.append(f"Avg Exec Time:   {format_ns(perf_data['avg_execution_time_ns'])}")
                output_lines.append(f"Max Exec Time:   {format_ns(perf_data['max_execution_time_ns'])}")
                output_lines.append(f"Math Util (avg): {perf_data.get('avg_math_utilization', 0):.1f}%")

        click.echo("\n".join(output_lines))
