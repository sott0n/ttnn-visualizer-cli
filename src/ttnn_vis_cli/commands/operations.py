"""Operations commands - display operation information."""

import click

from ..data.profiler_db import ProfilerDB
from ..output.formatter import OutputFormat, OutputFormatter, format_ns


@click.command()
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.option(
    "--top",
    "-n",
    type=int,
    default=None,
    help="Show top N operations by duration",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=100,
    help="Maximum number of operations to display (default: 100)",
)
@click.pass_context
def operations(ctx: click.Context, profiler: str, top: int | None, limit: int) -> None:
    """Display operations list.

    Shows all operations in the profiling report with their IDs,
    names, and execution durations. Use --top to show the slowest
    operations.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        if top:
            ops = db.get_operations(limit=top, order_by_duration=True)
            title = f"Top {top} Operations by Duration"
        else:
            ops = db.get_operations(limit=limit)
            title = "Operations"
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not ops:
        click.echo("No operations found")
        return

    if format_type == OutputFormat.JSON:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data, title=title))
    elif format_type == OutputFormat.CSV:
        data = [op.to_dict() for op in ops]
        click.echo(formatter.format_output(data))
    else:
        # Table format
        headers = ["ID", "Name", "Duration", "Device"]
        rows = []
        for op in ops:
            duration_str = format_ns(op.duration) if op.duration else "-"
            rows.append({
                "ID": op.id,
                "Name": op.name,
                "Duration": duration_str,
                "Device": op.device_id if op.device_id is not None else "-",
            })
        click.echo(formatter.format_output(rows, headers=headers, title=title))


@click.command()
@click.argument("operation_id", type=int)
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.pass_context
def operation(ctx: click.Context, operation_id: int, profiler: str) -> None:
    """Display detailed information about a specific operation.

    Shows operation details including arguments, input/output tensors,
    and associated buffers.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        op = db.get_operation(operation_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not op:
        raise click.ClickException(f"Operation {operation_id} not found")

    # Get related data
    arguments = db.get_operation_arguments(operation_id)
    input_tensors = db.get_input_tensors(operation_id)
    output_tensors = db.get_output_tensors(operation_id)
    buffers = db.get_buffers(operation_id=operation_id)

    # Get stack trace if available
    stack_trace = None
    if op.stack_trace_id:
        stack_trace = db.get_stack_trace(op.stack_trace_id)

    if format_type == OutputFormat.JSON:
        data = {
            "operation": op.to_dict(),
            "arguments": [arg.to_dict() for arg in arguments],
            "input_tensors": [t.to_dict() for t in input_tensors],
            "output_tensors": [t.to_dict() for t in output_tensors],
            "buffers": [b.to_dict() for b in buffers],
            "stack_trace": stack_trace,
        }
        click.echo(formatter.format_output(data, title=f"Operation {operation_id}"))
    else:
        # Table format
        output_lines = []
        output_lines.append(f"Operation {operation_id}: {op.name}")
        output_lines.append("=" * (len(f"Operation {operation_id}: {op.name}")))
        output_lines.append("")
        output_lines.append(f"Duration: {format_ns(op.duration) if op.duration else 'N/A'}")
        output_lines.append(f"Device:   {op.device_id if op.device_id is not None else 'N/A'}")

        if arguments:
            output_lines.append("")
            output_lines.append("Arguments:")
            for arg in arguments:
                value = arg.value
                if len(value) > 60:
                    value = value[:57] + "..."
                output_lines.append(f"  {arg.name}: {value}")

        if input_tensors:
            output_lines.append("")
            output_lines.append("Input Tensors:")
            for t in input_tensors:
                output_lines.append(f"  [{t.id}] {t.shape} {t.dtype} {t.layout}")

        if output_tensors:
            output_lines.append("")
            output_lines.append("Output Tensors:")
            for t in output_tensors:
                output_lines.append(f"  [{t.id}] {t.shape} {t.dtype} {t.layout}")

        if buffers:
            output_lines.append("")
            output_lines.append("Buffers:")
            for b in buffers:
                output_lines.append(f"  [{b.id}] {b.buffer_type.value} addr=0x{b.address:x} size={b.max_size:,}")

        if stack_trace:
            output_lines.append("")
            output_lines.append("Stack Trace:")
            for line in stack_trace.split("\n")[:10]:  # Limit to 10 lines
                output_lines.append(f"  {line}")
            if stack_trace.count("\n") > 10:
                output_lines.append("  ...")

        click.echo("\n".join(output_lines))
