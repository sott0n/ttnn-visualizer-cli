"""Utility functions for TUI screens."""

from __future__ import annotations


def escape_markup(text: str) -> str:
    """Escape all square brackets for Rich markup.

    Rich's escape() only escapes markup-like patterns (e.g., [bold]).
    Textual's visualize treats more patterns as markup, so we need to
    escape ALL square brackets.

    Args:
        text: Text to escape.

    Returns:
        Text with all square brackets escaped.
    """
    return text.replace("[", "\\[").replace("]", "\\]")


def format_time_ns(time_ns: float | None) -> str:
    """Format time in nanoseconds to human readable string.

    Args:
        time_ns: Time in nanoseconds, or None.

    Returns:
        Human readable time string.
    """
    if time_ns is None:
        return "-"
    if time_ns == 0:
        return "0 ns"
    if time_ns < 1000:
        return f"{time_ns:.0f} ns"
    if time_ns < 1_000_000:
        return f"{time_ns / 1000:.2f} µs"
    return f"{time_ns / 1_000_000:.2f} ms"
