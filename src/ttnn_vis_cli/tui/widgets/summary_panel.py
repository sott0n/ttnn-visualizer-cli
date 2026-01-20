"""Summary panel widget for displaying key-value pairs."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class SummaryPanel(Widget):
    """A panel displaying a title and key-value summary items."""

    DEFAULT_CSS = """
    SummaryPanel {
        border: solid $primary;
        padding: 1 2;
        height: auto;
        min-height: 7;
    }

    SummaryPanel .summary-title {
        text-style: bold;
        color: $text;
    }

    SummaryPanel .summary-item {
        color: $text;
    }
    """

    def __init__(
        self,
        title: str,
        items: dict[str, str],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the summary panel.

        Args:
            title: The panel title.
            items: Dictionary of label -> value pairs to display.
            name: Widget name.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._title = title
        self._items = items

    def compose(self) -> ComposeResult:
        """Compose the panel content."""
        yield Static(self._title, classes="summary-title")
        for label, value in self._items.items():
            yield Static(f"{label}: {value}", classes="summary-item")
