"""Main TUI application for TTNN Visualizer."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .screens import DashboardScreen, OperationsScreen, PerformanceScreen, TensorsScreen


class TTNNVisualizerApp(App):
    """TTNN Visualizer TUI Application."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "TTNN Visualizer"
    SUB_TITLE = "TUI"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "switch_tab('dashboard')", "Dashboard", show=False),
        Binding("o", "switch_tab('operations')", "Operations", show=False),
        Binding("t", "switch_tab('tensors')", "Tensors", show=False),
        Binding("p", "switch_tab('performance')", "Performance", show=False),
        Binding("?", "show_help", "Help"),
    ]

    def __init__(
        self,
        profiler_db: Path | None = None,
        perf_data: Path | None = None,
    ):
        """Initialize the TUI application.

        Args:
            profiler_db: Path to the profiler SQLite database.
            perf_data: Path to the performance data directory.
        """
        super().__init__()
        self.profiler_db = profiler_db
        self.perf_data = perf_data

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield DashboardScreen(
                    profiler_db=self.profiler_db,
                    perf_data=self.perf_data,
                )
            with TabPane("Operations", id="operations"):
                yield OperationsScreen(profiler_db=self.profiler_db)
            with TabPane("Tensors", id="tensors"):
                yield TensorsScreen(profiler_db=self.profiler_db)
            with TabPane("Performance", id="performance"):
                yield PerformanceScreen(perf_data=self.perf_data)
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab.

        Args:
            tab_id: The ID of the tab to switch to.
        """
        valid_tabs = {"dashboard", "operations", "tensors", "performance"}
        if tab_id not in valid_tabs:
            self.notify(f"Unknown tab: {tab_id}", severity="warning")
            return
        tabbed_content = self.query_one(TabbedContent)
        tabbed_content.active = tab_id

    def action_show_help(self) -> None:
        """Show help dialog."""
        self.notify(
            "Navigation:\n"
            "  d - Dashboard tab\n"
            "  o - Operations tab\n"
            "  t - Tensors tab\n"
            "  p - Performance tab\n"
            "  Tab - Next tab\n"
            "\n"
            "Table:\n"
            "  ↑/↓ - Navigate rows\n"
            "  Enter - Select row\n"
            "\n"
            "General:\n"
            "  ? - This help\n"
            "  q - Quit",
            title="Help",
            timeout=10,
        )


def run_tui(
    profiler_db: Path | None = None,
    perf_data: Path | None = None,
) -> None:
    """Run the TUI application.

    Args:
        profiler_db: Path to the profiler SQLite database.
        perf_data: Path to the performance data directory.
    """
    app = TTNNVisualizerApp(profiler_db=profiler_db, perf_data=perf_data)
    app.run()
