"""Memory commands - display memory and buffer information."""

import click

from ..data.models import BufferType
from ..data.profiler_db import ProfilerDB
from ..output.formatter import OutputFormat, OutputFormatter, format_bytes


@click.command()
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.pass_context
def memory(ctx: click.Context, profiler: str) -> None:
    """Display memory usage summary.

    Shows L1 and DRAM memory usage including total size,
    used size, and usage percentage.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        summary = db.get_memory_summary()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(summary.to_dict(), title="Memory Summary"))
    else:
        # Table format
        output_lines = []
        output_lines.append("Memory Usage Summary")
        output_lines.append("=" * 20)
        output_lines.append("")
        output_lines.append("L1 Memory:")
        output_lines.append(f"  Used:         {format_bytes(summary.l1_used)}")
        output_lines.append(f"  Total:        {format_bytes(summary.l1_total)}")
        output_lines.append(f"  Usage:        {summary.l1_usage_percent:.1f}%")
        output_lines.append(f"  Buffer Count: {summary.l1_buffer_count:,}")
        output_lines.append("")
        output_lines.append("DRAM Memory:")
        output_lines.append(f"  Used:         {format_bytes(summary.dram_used)}")
        output_lines.append(f"  Total:        {format_bytes(summary.dram_total)}")
        output_lines.append(f"  Usage:        {summary.dram_usage_percent:.1f}%")
        output_lines.append(f"  Buffer Count: {summary.dram_buffer_count:,}")

        click.echo("\n".join(output_lines))


@click.command()
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.option(
    "--type",
    "-t",
    "buffer_type",
    type=click.Choice(["L1", "DRAM", "L1_SMALL", "TRACE", "SYSTEM_MEMORY"]),
    default=None,
    help="Filter by buffer type",
)
@click.option(
    "--operation",
    "-o",
    "operation_id",
    type=int,
    default=None,
    help="Filter by operation ID",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=100,
    help="Maximum number of buffers to display (default: 100)",
)
@click.pass_context
def buffers(
    ctx: click.Context,
    profiler: str,
    buffer_type: str | None,
    operation_id: int | None,
    limit: int,
) -> None:
    """Display buffer list.

    Shows buffers in the profiling report with their IDs,
    addresses, sizes, and types. Can be filtered by buffer type
    or operation ID.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)

        # Convert string buffer type to enum
        type_filter = None
        if buffer_type:
            type_filter = BufferType[buffer_type]

        buffer_list = db.get_buffers(
            buffer_type=type_filter,
            operation_id=operation_id,
            limit=limit,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not buffer_list:
        click.echo("No buffers found")
        return

    title = "Buffers"
    if buffer_type:
        title += f" (type={buffer_type})"
    if operation_id is not None:
        title += f" (operation={operation_id})"

    if format_type == OutputFormat.JSON:
        data = [b.to_dict() for b in buffer_list]
        click.echo(formatter.format_output(data, title=title))
    elif format_type == OutputFormat.CSV:
        data = [b.to_dict() for b in buffer_list]
        click.echo(formatter.format_output(data))
    else:
        # Table format
        headers = ["ID", "Type", "Address", "Size", "Device", "Operation"]
        rows = []
        for b in buffer_list:
            rows.append({
                "ID": b.id,
                "Type": b.buffer_type.value,
                "Address": f"0x{b.address:x}",
                "Size": format_bytes(b.max_size),
                "Device": b.device_id,
                "Operation": b.operation_id if b.operation_id is not None else "-",
            })
        click.echo(formatter.format_output(rows, headers=headers, title=title))
