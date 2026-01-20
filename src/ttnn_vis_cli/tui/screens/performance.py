"""Performance analysis screen for TUI."""

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
    from ttnn_vis_cli.data.models import OperationPerf
    from ttnn_vis_cli.data.perf_analysis import AnalysisSummary, PerfAnalyzer
    from ttnn_vis_cli.data.perf_csv import PerfCSV

logger = logging.getLogger(__name__)

# Display constants
MAX_OP_CODE_LENGTH = 30
MAX_SHAPES_LENGTH = 50


def format_percent(value: float) -> str:
    """Format percentage value.

    Args:
        value: Percentage value.

    Returns:
        Formatted percentage string.
    """
    if value == 0:
        return "-"
    return f"{value:.1f}%"


class PerformanceScreen(Container):
    """Performance analysis screen with DataTable."""

    def __init__(
        self,
        perf_data: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the performance screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._perf_data_path = perf_data
        self._perf_csv: PerfCSV | None = None
        self._analyzer: PerfAnalyzer | None = None
        self._operations: list[OperationPerf] = []

    def compose(self) -> ComposeResult:
        """Compose the performance screen layout."""
        with Vertical():
            yield Static("[bold]Performance Analysis[/bold]", markup=True, classes="section-title")
            # Summary panel
            yield Container(
                Static("Loading...", id="performance-summary-content"),
                id="performance-summary",
                classes="performance-summary-panel",
            )
            with Horizontal(classes="performance-layout"):
                with Container(classes="performance-table-container"):
                    yield DataTable(id="performance-table", cursor_type="row")
                with Container(id="performance-detail", classes="detail-panel"):
                    yield Static("Select an operation to view details", id="performance-detail-content")

    def on_mount(self) -> None:
        """Load data when mounted."""
        self._init_perf_data()
        self._load_operations()

    def on_show(self) -> None:
        """Focus the table when shown."""
        try:
            table = self.query_one("#performance-table", DataTable)
            table.focus()
        except NoMatches:
            # Table not yet mounted or not available
            pass

    def _init_perf_data(self) -> None:
        """Initialize the performance data sources."""
        if not self._perf_data_path:
            return

        try:
            from ttnn_vis_cli.data.perf_analysis import PerfAnalyzer
            from ttnn_vis_cli.data.perf_csv import PerfCSV

            self._perf_csv = PerfCSV(self._perf_data_path)
            if self._perf_csv.is_valid():
                self._operations = self._perf_csv.get_operations()
                self._analyzer = PerfAnalyzer(self._operations)
        except Exception as e:
            logger.exception("Failed to initialize performance data")
            summary = self.query_one("#performance-summary-content", Static)
            summary.update(f"[red]Error initializing data: {e}[/red]")

    def _load_operations(self) -> None:
        """Load operations from the performance data."""
        summary_widget = self.query_one("#performance-summary-content", Static)
        detail = self.query_one("#performance-detail-content", Static)

        if not self._perf_csv or not self._perf_csv.is_valid():
            summary_widget.update("[yellow]No performance data configured[/yellow]")
            detail.update("[yellow]No performance data configured[/yellow]")
            return

        try:
            # Update summary panel
            if self._analyzer:
                summary = self._analyzer.get_summary()
                summary_widget.update(self._format_summary(summary))

            # Populate table
            table = self.query_one("#performance-table", DataTable)
            table.add_columns("ID", "Op Code", "Time", "FPU%", "Bound", "Cores")

            for idx, op in enumerate(self._operations):
                op_code = op.op_code[:MAX_OP_CODE_LENGTH] if len(op.op_code) > MAX_OP_CODE_LENGTH else op.op_code
                call_id = str(op.global_call_count) if op.global_call_count is not None else str(idx + 1)

                table.add_row(
                    call_id,
                    escape_markup(op_code),
                    format_time_ns(op.execution_time_ns),
                    format_percent(op.fpu_util_percent),
                    escape_markup(op.bound) if op.bound else "-",
                    str(op.core_count) if op.core_count > 0 else "-",
                    key=str(idx),
                )

        except Exception as e:
            logger.exception("Failed to load performance data")
            summary_widget.update(f"[red]Error loading data: {e}[/red]")

    def _format_summary(self, summary: AnalysisSummary) -> str:
        """Format the analysis summary for display.

        Args:
            summary: Analysis summary data.

        Returns:
            Formatted summary string.
        """
        total_time_ms = summary.total_device_time_ns / 1_000_000
        gap_time_ms = summary.total_op_to_op_gap_ns / 1_000_000

        lines = [
            f"[cyan]Total:[/cyan] {summary.total_operations} ops  "
            f"[cyan]Compute:[/cyan] {summary.compute_bound_count}  "
            f"[cyan]Memory:[/cyan] {summary.memory_bound_count}  "
            f"[cyan]Balanced:[/cyan] {summary.balanced_count}",
            f"[cyan]Device Time:[/cyan] {total_time_ms:.2f} ms  "
            f"[cyan]Gap Time:[/cyan] {gap_time_ms:.2f} ms  "
            f"[cyan]Avg FPU:[/cyan] {summary.avg_fpu_util:.1f}%  "
            f"[cyan]Avg DRAM:[/cyan] {summary.avg_dram_util:.1f}%",
        ]

        # Add top op codes if available
        if summary.top_op_codes:
            top_ops = [f"{op[0]} ({op[3]:.1f}%)" for op in summary.top_op_codes[:3]]
            lines.append(f"[cyan]Top Ops:[/cyan] {', '.join(top_ops)}")

        return "\n".join(lines)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the performance table."""
        if event.row_key is None or event.row_key.value is None:
            return

        op_idx = int(event.row_key.value)
        self._show_operation_detail(op_idx)

    def _show_operation_detail(self, op_idx: int) -> None:
        """Show details for the selected operation.

        Args:
            op_idx: Index of the operation in the operations list.
        """
        detail = self.query_one("#performance-detail-content", Static)

        if op_idx < 0 or op_idx >= len(self._operations):
            detail.update(f"[red]Operation {op_idx} not found[/red]")
            return

        try:
            op = self._operations[op_idx]

            # Build detail content (escape user data to prevent markup errors)
            call_id = op.global_call_count if op.global_call_count is not None else op_idx + 1
            lines = [
                f"[bold]Operation {call_id}[/bold]",
                "",
                f"[cyan]Op Code:[/cyan] {escape_markup(op.op_code)}",
                f"[cyan]Op Name:[/cyan] {escape_markup(op.op_name)}",
                f"[cyan]Device Time:[/cyan] {format_time_ns(op.execution_time_ns)}",
            ]

            if op.host_time_ns > 0:
                lines.append(f"[cyan]Host Time:[/cyan] {format_time_ns(op.host_time_ns)}")

            if op.op_to_op_gap_ns > 0:
                lines.append(f"[cyan]Op-to-Op Gap:[/cyan] {format_time_ns(op.op_to_op_gap_ns)}")

            lines.append("")
            lines.append("[cyan]Performance Metrics:[/cyan]")

            if op.core_count > 0:
                lines.append(f"  • Cores: {op.core_count}")

            if op.bound:
                lines.append(f"  • Bound: {escape_markup(op.bound)}")

            if op.fpu_util_percent > 0:
                lines.append(f"  • FPU Util: {op.fpu_util_percent:.1f}%")

            if op.dram_bw_util_percent > 0:
                lines.append(f"  • DRAM BW Util: {op.dram_bw_util_percent:.1f}%")

            # Performance model details
            if op.pm_ideal_ns is not None:
                lines.append("")
                lines.append("[cyan]Performance Model:[/cyan]")
                lines.append(f"  • Ideal: {format_time_ns(op.pm_ideal_ns)}")
                if op.pm_compute_ns is not None:
                    lines.append(f"  • Compute: {format_time_ns(op.pm_compute_ns)}")
                if op.pm_bandwidth_ns is not None:
                    lines.append(f"  • Bandwidth: {format_time_ns(op.pm_bandwidth_ns)}")
                # Calculate efficiency
                if op.execution_time_ns > 0 and op.pm_ideal_ns > 0:
                    efficiency = (op.pm_ideal_ns / op.execution_time_ns) * 100
                    lines.append(f"  • Efficiency: {efficiency:.1f}%")

            # Memory info
            if op.buffer_type or op.layout:
                lines.append("")
                lines.append("[cyan]Memory:[/cyan]")
                if op.buffer_type:
                    lines.append(f"  • Buffer Type: {escape_markup(op.buffer_type)}")
                if op.layout:
                    lines.append(f"  • Layout: {escape_markup(op.layout)}")

            # Shapes
            if op.input_shapes or op.output_shapes:
                lines.append("")
                lines.append("[cyan]Shapes:[/cyan]")
                if op.input_shapes:
                    shapes = op.input_shapes[:MAX_SHAPES_LENGTH]
                    if len(op.input_shapes) > MAX_SHAPES_LENGTH:
                        shapes += "..."
                    lines.append(f"  • Input: {escape_markup(shapes)}")
                if op.output_shapes:
                    shapes = op.output_shapes[:MAX_SHAPES_LENGTH]
                    if len(op.output_shapes) > MAX_SHAPES_LENGTH:
                        shapes += "..."
                    lines.append(f"  • Output: {escape_markup(shapes)}")

            detail.update("\n".join(lines))

        except Exception as e:
            logger.exception("Failed to load operation detail")
            detail.update(f"[red]Error: {e}[/red]")
