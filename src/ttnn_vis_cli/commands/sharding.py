"""Sharding analysis commands for TTNN Visualizer CLI."""

import csv
import io
from pathlib import Path
from typing import Optional

import click

from ..data.profiler_db import ProfilerDB
from ..data.sharding_analysis import (
    ShardingAnalyzer,
    detect_reshards,
)
from ..output.formatter import OutputFormat


def get_format(ctx: click.Context) -> OutputFormat:
    """Get output format from context."""
    format_str = ctx.obj.get("format", "table") if ctx.obj else "table"
    return OutputFormat(format_str)


@click.group()
@click.option(
    "--profiler",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to profiler SQLite database (db.sqlite)",
)
@click.pass_context
def sharding(ctx: click.Context, profiler: Path) -> None:
    """Sharding analysis commands.

    Analyze tensor sharding strategies for optimization.
    """
    ctx.ensure_object(dict)
    ctx.obj["profiler"] = profiler


@sharding.command(name="summary")
@click.pass_context
def sharding_summary(ctx: click.Context) -> None:
    """Show sharding strategy summary with recommendations.

    Example:
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite summary
    """
    profiler_path = ctx.obj["profiler"]
    output_format = get_format(ctx)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()
    operations_with_tensors = db.get_operations_with_tensors()

    analyzer = ShardingAnalyzer(tensors)
    reshards = detect_reshards(operations_with_tensors)
    reshard_count = sum(1 for r in reshards if r.has_reshard)

    summary = analyzer.get_summary(reshard_count=reshard_count)

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps(summary.to_dict(), indent=2))
    else:
        click.echo("=== Sharding Analysis Summary ===\n")
        click.echo(f"Total Tensors: {summary.total_tensors}")
        click.echo("")
        click.echo("Strategy Distribution:")
        click.echo(f"  HEIGHT_SHARDED: {summary.height_sharded_count}")
        click.echo(f"  WIDTH_SHARDED:  {summary.width_sharded_count}")
        click.echo(f"  BLOCK_SHARDED:  {summary.block_sharded_count}")
        click.echo(f"  INTERLEAVED:    {summary.interleaved_count}")
        click.echo(f"  SINGLE_BANK:    {summary.single_bank_count}")
        click.echo(f"  UNKNOWN:        {summary.unknown_count}")
        click.echo("")
        click.echo(f"Sharded:     {summary.sharded_percent:.1f}%")
        click.echo(f"Interleaved: {summary.interleaved_percent:.1f}%")
        click.echo(f"Reshard Ops: {summary.reshard_count}")
        click.echo("")
        click.echo("Recommendations:")
        for rec in summary.recommendations:
            click.echo(f"  - {rec}")


@sharding.command(name="distribution")
@click.pass_context
def sharding_distribution(ctx: click.Context) -> None:
    """Show sharding strategy distribution.

    Example:
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite distribution
    """
    profiler_path = ctx.obj["profiler"]
    output_format = get_format(ctx)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()

    analyzer = ShardingAnalyzer(tensors)
    distribution = analyzer.get_distribution()

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([d.to_dict() for d in distribution], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["strategy", "count", "percent", "l1_count", "dram_count"])
        for d in distribution:
            writer.writerow([d.strategy, d.count, f"{d.percent:.1f}", d.l1_count, d.dram_count])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        headers = ["Strategy", "Count", "Percent", "L1", "DRAM"]
        rows = [
            [d.strategy, d.count, f"{d.percent:.1f}%", d.l1_count, d.dram_count]
            for d in distribution
        ]
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))


@sharding.command(name="tensors")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(
        ["HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED", "INTERLEAVED", "SINGLE_BANK"],
        case_sensitive=False,
    ),
    help="Filter by sharding strategy",
)
@click.option(
    "--buffer",
    "-b",
    type=click.Choice(["L1", "DRAM"], case_sensitive=False),
    help="Filter by buffer type",
)
@click.option("--top", "-n", type=int, default=20, help="Number of tensors to show")
@click.pass_context
def sharding_tensors(
    ctx: click.Context,
    strategy: Optional[str],
    buffer: Optional[str],
    top: int,
) -> None:
    """List tensors with their sharding information.

    Example:
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite tensors
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite tensors --strategy HEIGHT_SHARDED
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite tensors --buffer L1 --top 10
    """
    profiler_path = ctx.obj["profiler"]
    output_format = get_format(ctx)

    db = ProfilerDB(profiler_path)
    tensors = db.get_tensors()

    analyzer = ShardingAnalyzer(tensors)
    results = analyzer.get_all_tensor_shardings(
        strategy_filter=strategy,
        buffer_filter=buffer,
        limit=top,
    )

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["tensor_id", "shape", "dtype", "layout", "buffer_type", "sharding_strategy"])
        for r in results:
            writer.writerow([r.tensor_id, r.shape, r.dtype, r.layout, r.buffer_type, r.sharding_strategy])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        headers = ["ID", "Shape", "DType", "Layout", "Buffer", "Sharding"]
        rows = [
            [
                r.tensor_id,
                r.shape[:30] + "..." if len(r.shape) > 30 else r.shape,
                r.dtype,
                r.layout,
                r.buffer_type,
                r.sharding_strategy,
            ]
            for r in results
        ]
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))

        if len(results) == top:
            click.echo(f"\n(Showing top {top} results, use --top to see more)")


@sharding.command(name="reshards")
@click.option("--top", "-n", type=int, default=20, help="Number of reshards to show")
@click.pass_context
def sharding_reshards(ctx: click.Context, top: int) -> None:
    """Detect reshard operations between consecutive operations.

    Example:
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite reshards
    """
    profiler_path = ctx.obj["profiler"]
    output_format = get_format(ctx)

    db = ProfilerDB(profiler_path)
    operations_with_tensors = db.get_operations_with_tensors()

    reshards = detect_reshards(operations_with_tensors)

    # Filter to only show operations with reshards
    reshards_only = [r for r in reshards if r.has_reshard]

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([r.to_dict() for r in reshards_only[:top]], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["operation_id", "operation_name", "reshard_detail"])
        for r in reshards_only[:top]:
            writer.writerow([r.operation_id, r.operation_name, r.reshard_detail or ""])
        click.echo(output.getvalue().rstrip())
    else:
        if not reshards_only:
            click.echo("No reshard operations detected.")
            click.echo("\nThis is good! Consistent sharding minimizes overhead.")
            return

        from tabulate import tabulate

        headers = ["Op ID", "Operation", "Reshard"]
        rows = [
            [
                r.operation_id,
                r.operation_name[:40] + "..." if len(r.operation_name) > 40 else r.operation_name,
                r.reshard_detail or "",
            ]
            for r in reshards_only[:top]
        ]
        click.echo(f"Detected {len(reshards_only)} reshard operations:\n")
        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))

        if len(reshards_only) > top:
            click.echo(f"\n(Showing top {top} results, use --top to see more)")

        click.echo("\nRecommendation: Consider using consistent sharding strategy")
        click.echo("to minimize reshard overhead.")


@sharding.command(name="operations")
@click.option("--top", "-n", type=int, default=20, help="Number of operations to show")
@click.pass_context
def sharding_operations(ctx: click.Context, top: int) -> None:
    """Show sharding information per operation.

    Example:
        ttnn-vis-cli sharding -p samples/profiler/db.sqlite operations
    """
    profiler_path = ctx.obj["profiler"]
    output_format = get_format(ctx)

    db = ProfilerDB(profiler_path)
    operations_with_tensors = db.get_operations_with_tensors(limit=top)

    reshards = detect_reshards(operations_with_tensors)

    if output_format == OutputFormat.JSON:
        import json

        click.echo(json.dumps([r.to_dict() for r in reshards], indent=2))
    elif output_format == OutputFormat.CSV:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["operation_id", "operation_name", "input_shardings", "output_shardings", "has_reshard"])
        for r in reshards:
            inputs = ";".join(r.input_shardings)
            outputs = ";".join(r.output_shardings)
            writer.writerow([r.operation_id, r.operation_name, inputs, outputs, r.has_reshard])
        click.echo(output.getvalue().rstrip())
    else:
        from tabulate import tabulate

        headers = ["Op ID", "Operation", "Input Shard", "Output Shard", "Reshard"]
        rows = []
        for r in reshards:
            input_shard = ", ".join(set(r.input_shardings)) if r.input_shardings else "-"
            output_shard = ", ".join(set(r.output_shardings)) if r.output_shardings else "-"
            reshard_mark = "YES" if r.has_reshard else ""

            rows.append([
                r.operation_id,
                r.operation_name[:30] + "..." if len(r.operation_name) > 30 else r.operation_name,
                input_shard[:20] if len(input_shard) > 20 else input_shard,
                output_shard[:20] if len(output_shard) > 20 else output_shard,
                reshard_mark,
            ])

        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
