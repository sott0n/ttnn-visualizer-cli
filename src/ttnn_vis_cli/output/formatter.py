"""Unified output formatting for TTNN CLI."""

import csv
import io
import json
from enum import Enum
from typing import Any


class OutputFormat(Enum):
    """Output format options."""

    TABLE = "table"
    JSON = "json"
    CSV = "csv"


class OutputFormatter:
    """Unified output formatter supporting multiple formats."""

    def __init__(self, format: OutputFormat = OutputFormat.TABLE):
        """Initialize formatter with specified format.

        Args:
            format: Output format (table, json, csv).
        """
        self.format = format

    def format_output(
        self,
        data: list[dict] | dict,
        headers: list[str] | None = None,
        title: str | None = None,
    ) -> str:
        """Format data according to the configured format.

        Args:
            data: Data to format (list of dicts or single dict).
            headers: Column headers for table/csv output.
            title: Optional title for the output.

        Returns:
            Formatted string output.
        """
        if self.format == OutputFormat.JSON:
            return self._format_json(data, title)
        elif self.format == OutputFormat.CSV:
            return self._format_csv(data, headers)
        else:
            return self._format_table(data, headers, title)

    def _format_json(self, data: list[dict] | dict, title: str | None = None) -> str:
        """Format data as JSON."""
        output = {"data": data}
        if title:
            output["title"] = title
        return json.dumps(output, indent=2, default=str)

    def _format_csv(
        self, data: list[dict] | dict, headers: list[str] | None = None
    ) -> str:
        """Format data as CSV."""
        if isinstance(data, dict):
            data = [data]

        if not data:
            return ""

        output = io.StringIO()
        if headers is None:
            headers = list(data[0].keys())

        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

        return output.getvalue()

    def _format_table(
        self,
        data: list[dict] | dict,
        headers: list[str] | None = None,
        title: str | None = None,
    ) -> str:
        """Format data as ASCII table."""
        from tabulate import tabulate

        if isinstance(data, dict):
            # Single dict: format as key-value pairs
            rows = [[k, self._format_value(v)] for k, v in data.items()]
            table = tabulate(rows, headers=["Field", "Value"], tablefmt="simple")
        else:
            # List of dicts: format as table
            if not data:
                return "No data"

            if headers is None:
                headers = list(data[0].keys())

            rows = [
                [self._format_value(row.get(h, "")) for h in headers] for row in data
            ]
            table = tabulate(rows, headers=headers, tablefmt="simple")

        if title:
            return f"{title}\n{'=' * len(title)}\n\n{table}"
        return table

    def _format_value(self, value: Any) -> str:
        """Format a single value for display."""
        if value is None:
            return "-"
        elif isinstance(value, float):
            if abs(value) >= 1_000_000:
                return f"{value:,.0f}"
            elif abs(value) >= 1:
                return f"{value:,.2f}"
            else:
                return f"{value:.6f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            return json.dumps(value)
        return str(value)


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_ns(ns: float) -> str:
    """Format nanoseconds to human-readable string."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    elif ns >= 1_000:
        return f"{ns / 1_000:.3f} µs"
    return f"{ns:.0f} ns"
