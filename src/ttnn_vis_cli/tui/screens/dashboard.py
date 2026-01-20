"""Dashboard screen for TUI."""

import logging
from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

logger = logging.getLogger(__name__)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string.

    Args:
        size: Size in bytes (can be negative).

    Returns:
        Human readable size string.
    """
    if size < 0:
        return f"-{format_bytes(abs(size))}"
    if size == 0:
        return "0 B"
    current_size = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if current_size < 1024:
            return f"{current_size:.1f} {unit}"
        current_size /= 1024
    return f"{current_size:.1f} TB"


@dataclass
class ProfilerData:
    """Cached profiler data."""

    operation_count: int = 0
    tensor_count: int = 0
    buffer_count: int = 0
    device_count: int = 0
    device_cores: int = 0
    device_x_cores: int = 0
    device_y_cores: int = 0
    device_l1_memory: int = 0
    l1_used: int = 0
    l1_total: int = 0
    dram_used: int = 0
    dram_total: int = 0
    error: str | None = None


@dataclass
class PerfData:
    """Cached performance data."""

    total_ops: int = 0
    total_time_ms: float = 0.0
    avg_fpu: float = 0.0
    top_op: str = "N/A"
    error: str | None = None


class DashboardScreen(Container):
    """Dashboard screen showing profiler and performance overview."""

    def __init__(
        self,
        profiler_db: Path | None = None,
        perf_data: Path | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the dashboard screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._profiler_db = profiler_db
        self._perf_data_path = perf_data
        self._profiler: ProfilerData | None = None
        self._perf: PerfData | None = None

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        # Load data once
        self._profiler = self._load_profiler_data()
        self._perf = self._load_perf_data()

        # Build content
        yield Static(
            f"[bold]Profiler Report[/bold]\n{self._format_profiler_content()}",
            markup=True,
        )
        yield Static("")
        yield Static(
            f"[bold]Performance[/bold]\n{self._format_perf_content()}",
            markup=True,
        )
        yield Static("")
        yield Static(
            f"[bold]Device 0[/bold]\n{self._format_device_content()}",
            markup=True,
        )
        yield Static("")
        yield Static(
            f"[bold]Memory Usage[/bold]\n{self._format_memory_content()}",
            markup=True,
        )

    def _load_profiler_data(self) -> ProfilerData:
        """Load all profiler data in a single pass."""
        if not self._profiler_db:
            return ProfilerData(error="No profiler data")

        try:
            from ttnn_vis_cli.data.profiler_db import ProfilerDB

            db = ProfilerDB(self._profiler_db)
            info = db.get_report_info()
            memory = db.get_memory_summary()

            # Calculate L1 total
            l1_total = memory.l1_total
            if l1_total == 0 and info.devices:
                l1_total = info.devices[0].total_l1_memory
            if l1_total == 0:
                l1_total = memory.l1_used if memory.l1_used > 0 else 1

            # Extract device info
            device_cores = 0
            device_x_cores = 0
            device_y_cores = 0
            device_l1_memory = 0
            if info.devices:
                dev = info.devices[0]
                device_cores = dev.num_compute_cores
                device_x_cores = dev.num_x_compute_cores
                device_y_cores = dev.num_y_compute_cores
                device_l1_memory = dev.total_l1_memory

            return ProfilerData(
                operation_count=info.operation_count,
                tensor_count=info.tensor_count,
                buffer_count=info.buffer_count,
                device_count=info.device_count,
                device_cores=device_cores,
                device_x_cores=device_x_cores,
                device_y_cores=device_y_cores,
                device_l1_memory=device_l1_memory,
                l1_used=memory.l1_used,
                l1_total=l1_total,
                dram_used=memory.dram_used,
                dram_total=memory.dram_total,
            )
        except Exception as e:
            logger.exception("Failed to load profiler data")
            return ProfilerData(error=str(e))

    def _load_perf_data(self) -> PerfData:
        """Load all performance data in a single pass."""
        if not self._perf_data_path:
            return PerfData(error="No performance data")

        try:
            from ttnn_vis_cli.data.perf_csv import PerfCSV

            perf = PerfCSV(self._perf_data_path)
            if not perf.is_valid():
                return PerfData(error="No CSV file found")

            summary = perf.get_summary()
            operations = perf.get_operations()

            fpu_values = [
                op.fpu_util_percent for op in operations if op.fpu_util_percent > 0
            ]
            avg_fpu = sum(fpu_values) / len(fpu_values) if fpu_values else 0

            top_ops = perf.get_top_operations(1)
            top_op_name = top_ops[0].op_code if top_ops else "N/A"

            return PerfData(
                total_ops=summary["total_operations"],
                total_time_ms=summary["total_execution_time_ms"],
                avg_fpu=avg_fpu,
                top_op=top_op_name,
            )
        except Exception as e:
            logger.exception("Failed to load performance data")
            return PerfData(error=str(e))

    def _format_profiler_content(self) -> str:
        """Format profiler content string."""
        if self._profiler is None or self._profiler.error:
            return self._profiler.error if self._profiler else "No data"

        return (
            f"Operations: {self._profiler.operation_count:,}\n"
            f"Tensors: {self._profiler.tensor_count:,}\n"
            f"Buffers: {self._profiler.buffer_count:,}\n"
            f"Devices: {self._profiler.device_count}"
        )

    def _format_perf_content(self) -> str:
        """Format performance content string."""
        if self._perf is None or self._perf.error:
            return self._perf.error if self._perf else "No data"

        return (
            f"Total Ops: {self._perf.total_ops}\n"
            f"Total Time: {self._perf.total_time_ms:.2f} ms\n"
            f"Avg FPU: {self._perf.avg_fpu:.1f}%\n"
            f"Top Op: {self._perf.top_op}"
        )

    def _format_device_content(self) -> str:
        """Format device content string."""
        if self._profiler is None or self._profiler.error:
            return "No device information"

        if self._profiler.device_cores == 0:
            return "No device information"

        return (
            f"Cores: {self._profiler.device_cores} "
            f"({self._profiler.device_x_cores}x{self._profiler.device_y_cores})  "
            f"L1: {format_bytes(self._profiler.device_l1_memory)}"
        )

    def _format_memory_content(self) -> str:
        """Format memory content string."""
        if self._profiler is None or self._profiler.error:
            return "No memory data"

        l1_pct = (
            (self._profiler.l1_used / self._profiler.l1_total) * 100
            if self._profiler.l1_total > 0
            else 0
        )

        # Calculate DRAM percentage if total is available
        if self._profiler.dram_total > 0:
            dram_pct = (self._profiler.dram_used / self._profiler.dram_total) * 100
            dram_str = (
                f"DRAM: {format_bytes(self._profiler.dram_used)} / "
                f"{format_bytes(self._profiler.dram_total)} ({dram_pct:.1f}%)"
            )
        else:
            dram_str = f"DRAM: {format_bytes(self._profiler.dram_used)}"

        return (
            f"L1:   {format_bytes(self._profiler.l1_used)} / "
            f"{format_bytes(self._profiler.l1_total)} ({l1_pct:.1f}%)\n"
            f"{dram_str}"
        )
