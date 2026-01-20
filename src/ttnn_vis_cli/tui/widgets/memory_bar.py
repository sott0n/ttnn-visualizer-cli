"""Memory usage bar widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static, ProgressBar


class MemoryBar(Widget):
    """A memory usage bar with label and percentage."""

    DEFAULT_CSS = """
    MemoryBar {
        height: 3;
        width: 100%;
        layout: horizontal;
    }

    MemoryBar .memory-label {
        width: 8;
        height: 1;
    }

    MemoryBar ProgressBar {
        width: 1fr;
        padding: 0 1;
    }

    MemoryBar .memory-percent {
        width: 8;
        height: 1;
        text-align: right;
    }
    """

    def __init__(
        self,
        label: str,
        used: int,
        total: int,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the memory bar.

        Args:
            label: Memory type label (e.g., "L1", "DRAM").
            used: Used memory in bytes.
            total: Total memory in bytes.
            name: Widget name.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._label = label
        self._used = used
        self._total = total

    @property
    def percentage(self) -> float:
        """Calculate usage percentage."""
        if self._total == 0:
            return 0.0
        return (self._used / self._total) * 100

    def compose(self) -> ComposeResult:
        """Compose the memory bar."""
        yield Static(self._label, classes="memory-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False)
        yield Static(f"{self.percentage:.0f}%", classes="memory-percent")

    def on_mount(self) -> None:
        """Set initial progress value."""
        progress_bar = self.query_one(ProgressBar)
        progress_bar.update(progress=self.percentage)
