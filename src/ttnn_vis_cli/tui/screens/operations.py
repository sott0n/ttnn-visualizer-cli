"""Operations browser screen for TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Static

from ..utils import escape_markup, format_time_ns

if TYPE_CHECKING:
    from ttnn_vis_cli.data.models import Operation
    from ttnn_vis_cli.data.profiler_db import ProfilerDB

logger = logging.getLogger(__name__)

# Display constants
MAX_NAME_LENGTH = 50
MAX_ARG_VALUE_LENGTH = 40
MAX_DISPLAYED_ARGS = 10


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
        self._profiler_db_path = profiler_db
        self._db: ProfilerDB | None = None
        self._operations: list[Operation] = []

    def compose(self) -> ComposeResult:
        """Compose the operations screen layout."""
        with Vertical():
            yield Static("[bold]Operations[/bold]", markup=True, classes="section-title")
            with Horizontal(classes="operations-layout"):
                with Container(classes="operations-table-container"):
                    yield DataTable(id="operations-table", cursor_type="row")
                with Container(id="operation-detail", classes="detail-panel"):
                    yield Static("Select an operation to view details", id="operation-detail-content")

    def on_mount(self) -> None:
        """Load data when mounted."""
        self._init_db()
        self._load_operations()

    def on_show(self) -> None:
        """Focus the table when shown."""
        try:
            table = self.query_one("#operations-table", DataTable)
            table.focus()
        except NoMatches:
            # Table not yet mounted or not available
            pass

    def _init_db(self) -> None:
        """Initialize the database connection."""
        if not self._profiler_db_path:
            return

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            self._db = ProfilerDB(self._profiler_db_path)
        except Exception as e:
            logger.exception("Failed to initialize database")
            detail = self.query_one("#operation-detail-content", Static)
            detail.update(f"[red]Error initializing database: {e}[/red]")

    def _load_operations(self) -> None:
        """Load operations from the profiler database."""
        detail = self.query_one("#operation-detail-content", Static)

        if not self._db:
            detail.update("[yellow]No profiler database configured[/yellow]")
            return

        try:
            self._operations = self._db.get_operations()

            table = self.query_one("#operations-table", DataTable)
            table.add_columns("ID", "Name", "Duration", "Device")

            for op in self._operations:
                device_str = str(op.device_id) if op.device_id is not None else "-"
                name = op.name[:MAX_NAME_LENGTH] if len(op.name) > MAX_NAME_LENGTH else op.name
                table.add_row(
                    str(op.id),
                    escape_markup(name),
                    format_time_ns(op.duration),
                    device_str,
                    key=str(op.id),
                )

        except Exception as e:
            logger.exception("Failed to load operations")
            detail.update(f"[red]Error loading operations: {e}[/red]")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the operations table."""
        if event.row_key is None or event.row_key.value is None:
            return

        operation_id = int(event.row_key.value)
        self._show_operation_detail(operation_id)

    def _show_operation_detail(self, operation_id: int) -> None:
        """Show details for the selected operation."""
        detail = self.query_one("#operation-detail-content", Static)

        if not self._db:
            detail.update("[yellow]No profiler database configured[/yellow]")
            return

        try:
            operation = self._db.get_operation(operation_id)

            if not operation:
                detail.update(f"[red]Operation {operation_id} not found[/red]")
                return

            # Get input and output tensors
            input_tensors = self._db.get_input_tensors(operation_id)
            output_tensors = self._db.get_output_tensors(operation_id)

            # Build detail content (escape user data to prevent markup errors)
            lines = [
                f"[bold]Operation {operation.id}[/bold]",
                "",
                f"[cyan]Name:[/cyan] {escape_markup(operation.name)}",
                f"[cyan]Duration:[/cyan] {format_time_ns(operation.duration)}",
                f"[cyan]Device:[/cyan] {operation.device_id if operation.device_id is not None else '-'}",
            ]

            if input_tensors:
                lines.append("")
                lines.append("[cyan]Input Tensors:[/cyan]")
                for t in input_tensors:
                    lines.append(f"  • Tensor {t.id}: {escape_markup(t.shape)} ({escape_markup(t.dtype)})")

            if output_tensors:
                lines.append("")
                lines.append("[cyan]Output Tensors:[/cyan]")
                for t in output_tensors:
                    lines.append(f"  • Tensor {t.id}: {escape_markup(t.shape)} ({escape_markup(t.dtype)})")

            # Get operation arguments if available
            args = self._db.get_operation_arguments(operation_id)
            if args:
                lines.append("")
                lines.append("[cyan]Arguments:[/cyan]")
                for arg in args[:MAX_DISPLAYED_ARGS]:
                    value = arg.value[:MAX_ARG_VALUE_LENGTH] if len(arg.value) > MAX_ARG_VALUE_LENGTH else arg.value
                    lines.append(f"  • {escape_markup(arg.name)}: {escape_markup(value)}")
                if len(args) > MAX_DISPLAYED_ARGS:
                    lines.append(f"  ... and {len(args) - MAX_DISPLAYED_ARGS} more")

            detail.update("\n".join(lines))

        except Exception as e:
            logger.exception("Failed to load operation detail")
            detail.update(f"[red]Error: {e}[/red]")
