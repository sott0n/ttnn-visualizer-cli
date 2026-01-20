"""Tensors browser screen for TUI."""

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Static

logger = logging.getLogger(__name__)


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
        self._profiler_db = profiler_db
        self._tensors: list = []
        self._selected_tensor_id: int | None = None

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
        self._load_tensors()

    def _load_tensors(self) -> None:
        """Load tensors from the profiler database."""
        if not self._profiler_db:
            return

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            self._tensors = db.get_tensors()

            table = self.query_one("#tensors-table", DataTable)
            table.add_columns("ID", "Shape", "Dtype", "Layout", "Memory")

            for t in self._tensors:
                # Truncate shape if too long
                shape = t.shape[:20] if len(t.shape) > 20 else t.shape
                # Determine memory type
                memory = t.buffer_type if t.buffer_type else "-"

                table.add_row(
                    str(t.id),
                    shape,
                    t.dtype,
                    t.layout,
                    memory,
                    key=str(t.id),
                )

        except Exception as e:
            logger.exception("Failed to load tensors")
            detail = self.query_one("#tensor-detail-content", Static)
            detail.update(f"[red]Error loading tensors: {e}[/red]")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the tensors table."""
        if event.row_key is None:
            return

        tensor_id = int(event.row_key.value)
        self._selected_tensor_id = tensor_id
        self._show_tensor_detail(tensor_id)

    def _show_tensor_detail(self, tensor_id: int) -> None:
        """Show details for the selected tensor."""
        if not self._profiler_db:
            return

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            tensor = db.get_tensor(tensor_id)

            if not tensor:
                detail = self.query_one("#tensor-detail-content", Static)
                detail.update(f"[red]Tensor {tensor_id} not found[/red]")
                return

            # Build detail content
            lines = [
                f"[bold]Tensor {tensor.id}[/bold]",
                "",
                f"[cyan]Shape:[/cyan] {tensor.shape}",
                f"[cyan]Dtype:[/cyan] {tensor.dtype}",
                f"[cyan]Layout:[/cyan] {tensor.layout}",
            ]

            if tensor.device_id is not None:
                lines.append(f"[cyan]Device:[/cyan] {tensor.device_id}")

            if tensor.buffer_type:
                lines.append(f"[cyan]Memory Type:[/cyan] {tensor.buffer_type}")

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
                if len(config) <= 100:
                    lines.append(f"  • Full: {config}")
                else:
                    lines.append(f"  • Full: {config[:100]}...")

            detail = self.query_one("#tensor-detail-content", Static)
            detail.update("\n".join(lines))

        except Exception as e:
            logger.exception("Failed to load tensor detail")
            detail = self.query_one("#tensor-detail-content", Static)
            detail.update(f"[red]Error: {e}[/red]")
