"""Operations browser screen for TUI."""

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Static

logger = logging.getLogger(__name__)


def format_duration(duration_ns: float | None) -> str:
    """Format duration in nanoseconds to human readable string.

    Args:
        duration_ns: Duration in nanoseconds.

    Returns:
        Human readable duration string.
    """
    if duration_ns is None:
        return "-"
    if duration_ns == 0:
        return "0 ns"
    if duration_ns < 1000:
        return f"{duration_ns:.0f} ns"
    if duration_ns < 1_000_000:
        return f"{duration_ns / 1000:.2f} µs"
    return f"{duration_ns / 1_000_000:.2f} ms"


class OperationsScreen(Container):
    """Operations browser screen with DataTable."""

    def __init__(
        self,
        profiler_db: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the operations screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._profiler_db = profiler_db
        self._operations: list = []
        self._selected_operation_id: int | None = None

    def compose(self) -> ComposeResult:
        """Compose the operations screen layout."""
        with Vertical():
            yield Static("[bold]Operations[/bold]", markup=True, classes="section-title")
            with Horizontal(classes="operations-layout"):
                with Container(classes="operations-table-container"):
                    yield DataTable(id="operations-table", cursor_type="row")
                with Container(id="operation-detail", classes="detail-panel"):
                    yield Static("Select an operation to view details", id="detail-content")

    def on_mount(self) -> None:
        """Load data when mounted."""
        self._load_operations()

    def _load_operations(self) -> None:
        """Load operations from the profiler database."""
        if not self._profiler_db:
            return

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            self._operations = db.get_operations()

            table = self.query_one("#operations-table", DataTable)
            table.add_columns("ID", "Name", "Duration", "Device")

            for op in self._operations:
                device_str = str(op.device_id) if op.device_id is not None else "-"
                table.add_row(
                    str(op.id),
                    op.name[:50] if len(op.name) > 50 else op.name,
                    format_duration(op.duration),
                    device_str,
                    key=str(op.id),
                )

        except Exception as e:
            logger.exception("Failed to load operations")
            detail = self.query_one("#detail-content", Static)
            detail.update(f"[red]Error loading operations: {e}[/red]")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the operations table."""
        if event.row_key is None:
            return

        operation_id = int(event.row_key.value)
        self._selected_operation_id = operation_id
        self._show_operation_detail(operation_id)

    def _show_operation_detail(self, operation_id: int) -> None:
        """Show details for the selected operation."""
        if not self._profiler_db:
            return

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            operation = db.get_operation(operation_id)

            if not operation:
                detail = self.query_one("#detail-content", Static)
                detail.update(f"[red]Operation {operation_id} not found[/red]")
                return

            # Get input and output tensors
            input_tensors = db.get_input_tensors(operation_id)
            output_tensors = db.get_output_tensors(operation_id)

            # Build detail content
            lines = [
                f"[bold]Operation {operation.id}[/bold]",
                "",
                f"[cyan]Name:[/cyan] {operation.name}",
                f"[cyan]Duration:[/cyan] {format_duration(operation.duration)}",
                f"[cyan]Device:[/cyan] {operation.device_id if operation.device_id is not None else '-'}",
            ]

            if input_tensors:
                lines.append("")
                lines.append("[cyan]Input Tensors:[/cyan]")
                for t in input_tensors:
                    lines.append(f"  • Tensor {t.id}: {t.shape} ({t.dtype})")

            if output_tensors:
                lines.append("")
                lines.append("[cyan]Output Tensors:[/cyan]")
                for t in output_tensors:
                    lines.append(f"  • Tensor {t.id}: {t.shape} ({t.dtype})")

            # Get operation arguments if available
            args = db.get_operation_arguments(operation_id)
            if args:
                lines.append("")
                lines.append("[cyan]Arguments:[/cyan]")
                for arg in args[:10]:  # Limit to first 10 args
                    value = arg.value[:40] if len(arg.value) > 40 else arg.value
                    lines.append(f"  • {arg.name}: {value}")
                if len(args) > 10:
                    lines.append(f"  ... and {len(args) - 10} more")

            detail = self.query_one("#detail-content", Static)
            detail.update("\n".join(lines))

        except Exception as e:
            logger.exception("Failed to load operation detail")
            detail = self.query_one("#detail-content", Static)
            detail.update(f"[red]Error: {e}[/red]")
