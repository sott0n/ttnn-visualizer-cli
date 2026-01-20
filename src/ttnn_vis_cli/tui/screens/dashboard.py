"""Dashboard screen for TUI."""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    if size == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class DashboardScreen(Container):
    """Dashboard screen showing profiler and performance overview."""

    def __init__(
        self,
        profiler_db: Optional[Path] = None,
        perf_data: Optional[Path] = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the dashboard screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._profiler_db = profiler_db
        self._perf_data = perf_data

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        # Build content strings
        profiler_content = self._get_profiler_content()
        perf_content = self._get_perf_content()
        device_content = self._get_device_content()
        memory_content = self._get_memory_content()

        yield Static(f"[bold]Profiler Report[/bold]\n{profiler_content}", markup=True)
        yield Static("")
        yield Static(f"[bold]Performance[/bold]\n{perf_content}", markup=True)
        yield Static("")
        yield Static(f"[bold]Device 0[/bold]\n{device_content}", markup=True)
        yield Static("")
        yield Static(f"[bold]Memory Usage[/bold]\n{memory_content}", markup=True)

    def _get_profiler_content(self) -> str:
        """Get profiler content string."""
        if not self._profiler_db:
            return "No profiler data"

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            info = db.get_report_info()

            return (
                f"Operations: {info.operation_count:,}\n"
                f"Tensors: {info.tensor_count:,}\n"
                f"Buffers: {info.buffer_count:,}\n"
                f"Devices: {info.device_count}"
            )
        except Exception as e:
            return f"Error: {e}"

    def _get_perf_content(self) -> str:
        """Get performance content string."""
        if not self._perf_data:
            return "No performance data"

        try:
            from ttnn_vis_cli.data.perf_csv import PerfCSV

            perf = PerfCSV(self._perf_data)
            if not perf.is_valid():
                return "No CSV file found"

            summary = perf.get_summary()
            operations = perf.get_operations()

            fpu_values = [op.fpu_util_percent for op in operations if op.fpu_util_percent > 0]
            avg_fpu = sum(fpu_values) / len(fpu_values) if fpu_values else 0

            top_ops = perf.get_top_operations(1)
            top_op_name = top_ops[0].op_code if top_ops else "N/A"

            return (
                f"Total Ops: {summary['total_operations']}\n"
                f"Total Time: {summary['total_execution_time_ms']:.2f} ms\n"
                f"Avg FPU: {avg_fpu:.1f}%\n"
                f"Top Op: {top_op_name}"
            )
        except Exception as e:
            return f"Error: {e}"

    def _get_device_content(self) -> str:
        """Get device content string."""
        if not self._profiler_db:
            return "No device information"

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            info = db.get_report_info()

            if info.devices:
                dev = info.devices[0]
                return (
                    f"Cores: {dev.num_compute_cores} ({dev.num_x_compute_cores}x{dev.num_y_compute_cores})  "
                    f"L1: {format_bytes(dev.total_l1_memory)}  "
                    f"Compute: {dev.num_compute_cores}"
                )
            return "No device information"
        except Exception as e:
            return f"Error: {e}"

    def _get_memory_content(self) -> str:
        """Get memory content string."""
        if not self._profiler_db:
            return "No memory data"

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            info = db.get_report_info()
            memory = db.get_memory_summary()

            l1_total = memory.l1_total
            if l1_total == 0 and info.devices:
                l1_total = info.devices[0].total_l1_memory
            if l1_total == 0:
                l1_total = memory.l1_used if memory.l1_used > 0 else 1

            l1_pct = (memory.l1_used / l1_total) * 100 if l1_total > 0 else 0
            dram_pct = 100.0  # DRAM is usually fully allocated

            return (
                f"L1:   {format_bytes(memory.l1_used)} / {format_bytes(l1_total)} ({l1_pct:.1f}%)\n"
                f"DRAM: {format_bytes(memory.dram_used)} ({dram_pct:.0f}%)"
            )
        except Exception as e:
            return f"Error: {e}"
