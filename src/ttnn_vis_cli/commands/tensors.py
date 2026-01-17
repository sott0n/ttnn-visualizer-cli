"""Tensors commands - display tensor information."""

import click

from ..data.profiler_db import ProfilerDB
from ..output.formatter import OutputFormat, OutputFormatter


@click.command()
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=100,
    help="Maximum number of tensors to display (default: 100)",
)
@click.pass_context
def tensors(ctx: click.Context, profiler: str, limit: int) -> None:
    """Display tensors list.

    Shows all tensors in the profiling report with their IDs,
    shapes, data types, and layouts.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        tensor_list = db.get_tensors(limit=limit)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not tensor_list:
        click.echo("No tensors found")
        return

    if format_type == OutputFormat.JSON:
        data = [t.to_dict() for t in tensor_list]
        click.echo(formatter.format_output(data, title="Tensors"))
    elif format_type == OutputFormat.CSV:
        data = [t.to_dict() for t in tensor_list]
        click.echo(formatter.format_output(data))
    else:
        # Table format
        headers = ["ID", "Shape", "Dtype", "Layout", "Memory Config", "Device"]
        rows = []
        for t in tensor_list:
            rows.append({
                "ID": t.id,
                "Shape": t.shape,
                "Dtype": t.dtype,
                "Layout": t.layout,
                "Memory Config": t.memory_config or "-",
                "Device": t.device_id if t.device_id is not None else "-",
            })
        click.echo(formatter.format_output(rows, headers=headers, title="Tensors"))


@click.command()
@click.argument("tensor_id", type=int)
@click.option(
    "--profiler",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to profiler database (db.sqlite)",
)
@click.pass_context
def tensor(ctx: click.Context, tensor_id: int, profiler: str) -> None:
    """Display detailed information about a specific tensor.

    Shows tensor details including shape, data type, layout,
    memory configuration, and buffer information.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        t = db.get_tensor(tensor_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not t:
        raise click.ClickException(f"Tensor {tensor_id} not found")

    if format_type == OutputFormat.JSON:
        click.echo(formatter.format_output(t.to_dict(), title=f"Tensor {tensor_id}"))
    else:
        # Table format
        output_lines = []
        output_lines.append(f"Tensor {tensor_id}")
        output_lines.append("=" * (len(f"Tensor {tensor_id}")))
        output_lines.append("")
        output_lines.append(f"Shape:         {t.shape}")
        output_lines.append(f"Data Type:     {t.dtype}")
        output_lines.append(f"Layout:        {t.layout}")
        output_lines.append(f"Memory Config: {t.memory_config or 'N/A'}")
        output_lines.append(f"Device:        {t.device_id if t.device_id is not None else 'N/A'}")
        output_lines.append(f"Address:       {f'0x{t.address:x}' if t.address else 'N/A'}")
        output_lines.append(f"Buffer Type:   {t.buffer_type or 'N/A'}")

        click.echo("\n".join(output_lines))
