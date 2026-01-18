"""L1 memory report command - display L1 memory layout for operations."""

import click

from ..data.profiler_db import ProfilerDB
from ..output.formatter import OutputFormat, OutputFormatter, format_bytes
from ..output.memory_map import format_l1_entry_table, render_memory_map


@click.command()
@click.argument("operation_id", type=int)
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.option(
    "--device",
    "-d",
    "device_id",
    type=int,
    default=None,
    help="Filter by device ID",
)
@click.option(
    "--hex/--no-hex",
    default=True,
    help="Show addresses in hexadecimal (default: hex)",
)
@click.option(
    "--previous/--no-previous",
    default=True,
    help="Show previous operation's L1 report (default: show)",
)
@click.pass_context
def l1_report(
    ctx: click.Context,
    operation_id: int,
    profiler: str,
    device_id: int | None,
    hex: bool,
    previous: bool,
) -> None:
    """Display L1 memory report for an operation.

    Shows Previous and Current L1 memory layout with tensor allocations.
    Displays memory map visualization and detailed tensor information including
    Address, Size, Shape, Data Type, and TensorMemoryLayout.

    OPERATION_ID is the ID of the operation to display the L1 report for.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)

        # Get the operation info
        operation = db.get_operation(operation_id)
        if not operation:
            raise click.ClickException(f"Operation {operation_id} not found")

        # Get device info for total L1 memory
        devices = db.get_devices()
        if not devices:
            raise click.ClickException("No device information found in database")

        # Use specified device or first device
        device = None
        if device_id is not None:
            device = next((d for d in devices if d.id == device_id), None)
            if not device:
                raise click.ClickException(f"Device {device_id} not found")
        else:
            device = devices[0]

        total_l1 = device.total_l1_for_tensors

        # Get current L1 report
        current_entries = db.get_l1_report(operation_id, device_id)

        # Get previous L1 report if requested
        previous_entries = []
        prev_operation = None
        if previous:
            previous_entries = db.get_previous_l1_report(operation_id, device_id)
            if previous_entries:
                # Get previous operation info
                prev_op_id = previous_entries[0].operation_id
                if prev_op_id:
                    prev_operation = db.get_operation(prev_op_id)

    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    # Format output
    if format_type == OutputFormat.JSON:
        output_data = {
            "operation_id": operation_id,
            "operation_name": operation.name,
            "device_id": device.id,
            "total_l1_memory": total_l1,
            "current_report": {
                "entries": [e.to_dict() for e in current_entries],
                "total_used": sum(e.size for e in current_entries),
                "entry_count": len(current_entries),
            },
        }
        if previous:
            prev_op_info = None
            if prev_operation:
                prev_op_info = {
                    "operation_id": prev_operation.id,
                    "operation_name": prev_operation.name,
                }
            output_data["previous_report"] = {
                "operation": prev_op_info,
                "entries": [e.to_dict() for e in previous_entries],
                "total_used": sum(e.size for e in previous_entries),
                "entry_count": len(previous_entries),
            }

        click.echo(formatter.format_output(output_data, title="L1 Memory Report"))
    else:
        # Table/text format
        output_lines = []

        # Header
        output_lines.append(f"L1 Memory Report - Operation {operation_id}: {operation.name}")
        output_lines.append("=" * 60)
        output_lines.append(f"Device: {device.id} | Total L1 for Tensors: {format_bytes(total_l1)}")
        output_lines.append("")

        # Previous L1 Report
        if previous:
            if prev_operation:
                output_lines.append(
                    f"Previous L1 Report (Operation {prev_operation.id}: {prev_operation.name}):"
                )
            else:
                output_lines.append("Previous L1 Report:")
            output_lines.append("-" * 50)

            if previous_entries:
                # Memory map
                output_lines.append("Memory Map:")
                memory_map_lines = render_memory_map(previous_entries, total_l1)
                output_lines.extend(memory_map_lines)
                output_lines.append("")

                # Entry table
                table_lines = format_l1_entry_table(previous_entries, show_hex=hex)
                output_lines.extend(table_lines)
            else:
                output_lines.append("No L1 buffers allocated")

            output_lines.append("")

        # Current L1 Report
        output_lines.append(f"Current L1 Report (Operation {operation_id}: {operation.name}):")
        output_lines.append("-" * 50)

        if current_entries:
            # Get previous addresses for marking new entries
            previous_addresses = {e.address for e in previous_entries}

            # Memory map
            output_lines.append("Memory Map:")
            memory_map_lines = render_memory_map(current_entries, total_l1)
            output_lines.extend(memory_map_lines)
            output_lines.append("")

            # Entry table with new markers
            table_lines = format_l1_entry_table(
                current_entries,
                show_hex=hex,
                mark_new=previous,
                previous_addresses=previous_addresses,
            )
            output_lines.extend(table_lines)
        else:
            output_lines.append("No L1 buffers allocated")

        click.echo("\n".join(output_lines))
