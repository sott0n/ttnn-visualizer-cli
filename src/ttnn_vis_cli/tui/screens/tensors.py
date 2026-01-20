"""Tensors browser screen for TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import DataTable, Static

from ..utils import escape_markup

if TYPE_CHECKING:
    from ttnn_vis_cli.data.models import Tensor
    from ttnn_vis_cli.data.profiler_db import ProfilerDB

logger = logging.getLogger(__name__)

# Display constants
MAX_SHAPE_LENGTH = 20
MAX_CONFIG_LENGTH = 100


def format_address(address: int | None) -> str:
    """Format memory address to hex string.

    Args:
        address: Memory address.

    Returns:
        Hex string or '-' if None.
    """
    if address is None:
        return "-"
    return f"0x{address:08X}"


class TensorsScreen(Container):
    """Tensors browser screen with DataTable."""

    def __init__(
        self,
        profiler_db: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the tensors screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._profiler_db_path = profiler_db
        self._db: ProfilerDB | None = None
        self._tensors: list[Tensor] = []

    def compose(self) -> ComposeResult:
        """Compose the tensors screen layout."""
        with Vertical():
            yield Static("[bold]Tensors[/bold]", markup=True, classes="section-title")
            with Horizontal(classes="tensors-layout"):
                with Container(classes="tensors-table-container"):
                    yield DataTable(id="tensors-table", cursor_type="row")
                with Container(id="tensor-detail", classes="detail-panel"):
                    yield Static("Select a tensor to view details", id="tensor-detail-content")

    def on_mount(self) -> None:
        """Load data when mounted."""
        self._init_db()
        self._load_tensors()

    def on_show(self) -> None:
        """Focus the table when shown."""
        try:
            table = self.query_one("#tensors-table", DataTable)
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
            detail = self.query_one("#tensor-detail-content", Static)
            detail.update(f"[red]Error initializing database: {e}[/red]")

    def _load_tensors(self) -> None:
        """Load tensors from the profiler database."""
        detail = self.query_one("#tensor-detail-content", Static)

        if not self._db:
            detail.update("[yellow]No profiler database configured[/yellow]")
            return

        try:
            self._tensors = self._db.get_tensors()

            table = self.query_one("#tensors-table", DataTable)
            table.add_columns("ID", "Shape", "Dtype", "Layout", "Memory")

            for t in self._tensors:
                # Truncate shape if too long
                shape = t.shape[:MAX_SHAPE_LENGTH] if len(t.shape) > MAX_SHAPE_LENGTH else t.shape
                # Determine memory type
                memory = t.buffer_type if t.buffer_type else "-"

                table.add_row(
                    str(t.id),
                    escape_markup(shape),
                    escape_markup(t.dtype),
                    escape_markup(t.layout),
                    escape_markup(memory),
                    key=str(t.id),
                )

        except Exception as e:
            logger.exception("Failed to load tensors")
            detail.update(f"[red]Error loading tensors: {e}[/red]")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the tensors table."""
        if event.row_key is None or event.row_key.value is None:
            return

        tensor_id = int(event.row_key.value)
        self._show_tensor_detail(tensor_id)

    def _show_tensor_detail(self, tensor_id: int) -> None:
        """Show details for the selected tensor."""
        detail = self.query_one("#tensor-detail-content", Static)

        if not self._db:
            detail.update("[yellow]No profiler database configured[/yellow]")
            return

        try:
            tensor = self._db.get_tensor(tensor_id)

            if not tensor:
                detail.update(f"[red]Tensor {tensor_id} not found[/red]")
                return

            # Build detail content (escape user data to prevent markup errors)
            lines = [
                f"[bold]Tensor {tensor.id}[/bold]",
                "",
                f"[cyan]Shape:[/cyan] {escape_markup(tensor.shape)}",
                f"[cyan]Dtype:[/cyan] {escape_markup(tensor.dtype)}",
                f"[cyan]Layout:[/cyan] {escape_markup(tensor.layout)}",
            ]

            if tensor.device_id is not None:
                lines.append(f"[cyan]Device:[/cyan] {tensor.device_id}")

            if tensor.buffer_type:
                lines.append(f"[cyan]Memory Type:[/cyan] {escape_markup(tensor.buffer_type)}")

            if tensor.address is not None:
                lines.append(f"[cyan]Address:[/cyan] {format_address(tensor.address)}")

            if tensor.memory_config:
                lines.append("")
                lines.append("[cyan]Memory Config:[/cyan]")
                # Parse and display memory config in a readable format
                config = tensor.memory_config
                # Try to extract key parts
                if "INTERLEAVED" in config:
                    lines.append("  • Layout: INTERLEAVED")
                elif "HEIGHT_SHARDED" in config:
                    lines.append("  • Layout: HEIGHT_SHARDED")
                elif "WIDTH_SHARDED" in config:
                    lines.append("  • Layout: WIDTH_SHARDED")
                elif "BLOCK_SHARDED" in config:
                    lines.append("  • Layout: BLOCK_SHARDED")

                # Show full config if it's not too long
                if len(config) <= MAX_CONFIG_LENGTH:
                    lines.append(f"  • Full: {escape_markup(config)}")
                else:
                    lines.append(f"  • Full: {escape_markup(config[:MAX_CONFIG_LENGTH])}...")

            detail.update("\n".join(lines))

        except Exception as e:
            logger.exception("Failed to load tensor detail")
            detail.update(f"[red]Error: {e}[/red]")
