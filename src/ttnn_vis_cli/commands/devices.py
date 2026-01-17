"""Devices command - display device information."""

import click

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
def devices(ctx: click.Context, profiler: str) -> None:
    """Display device information.

    Shows details about all devices including core configuration,
    memory sizes, and compute capabilities.
    """
    format_type = OutputFormat(ctx.obj.get("format", "table"))
    formatter = OutputFormatter(format_type)

    try:
        db = ProfilerDB(profiler)
        device_list = db.get_devices()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    if not device_list:
        click.echo("No devices found")
        return

    if format_type == OutputFormat.JSON:
        data = [d.to_dict() for d in device_list]
        click.echo(formatter.format_output(data, title="Devices"))
    elif format_type == OutputFormat.CSV:
        data = [d.to_dict() for d in device_list]
        click.echo(formatter.format_output(data))
    else:
        # Table format
        output_lines = ["Devices", "=" * 7, ""]

        for dev in device_list:
            output_lines.append(f"Device {dev.id}")
            output_lines.append("-" * 20)
            output_lines.append(f"  Architecture:       {dev.arch or 'N/A'}")
            output_lines.append(f"  Chip ID:            {dev.chip_id}")
            output_lines.append("")
            output_lines.append("  Core Configuration:")
            output_lines.append(f"    Total Cores:      {dev.total_cores} ({dev.num_x_cores}x{dev.num_y_cores})")
            output_lines.append(f"    Compute Cores:    {dev.total_compute_cores} ({dev.num_x_compute_cores}x{dev.num_y_compute_cores})")
            output_lines.append(f"    Num Compute:      {dev.num_compute_cores}")
            output_lines.append(f"    Num Storage:      {dev.num_storage_cores}")
            output_lines.append("")
            output_lines.append("  L1 Memory:")
            output_lines.append(f"    Worker L1 Size:   {format_bytes(dev.worker_l1_size)}")
            output_lines.append(f"    Total L1 Memory:  {format_bytes(dev.total_l1_memory)}")
            output_lines.append(f"    L1 for Tensors:   {format_bytes(dev.total_l1_for_tensors)}")
            output_lines.append(f"    L1 Num Banks:     {dev.l1_num_banks}")
            output_lines.append(f"    L1 Bank Size:     {format_bytes(dev.l1_bank_size)}")
            output_lines.append(f"    CB Limit:         {format_bytes(dev.cb_limit)}")
            output_lines.append("")

        click.echo("\n".join(output_lines))
