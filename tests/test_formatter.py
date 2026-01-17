"""Tests for output formatters."""

import json

from ttnn_vis_cli.output.formatter import (
    OutputFormat,
    OutputFormatter,
    format_bytes,
    format_ns,
)


def test_format_bytes():
    """Test format_bytes function."""
    assert format_bytes(100) == "100.0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1024 * 1024) == "1.0 MB"
    assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"


def test_format_ns():
    """Test format_ns function."""
    assert format_ns(100) == "100 ns"
    assert format_ns(1000) == "1.000 µs"
    assert format_ns(1000000) == "1.000 ms"
    assert format_ns(1000000000) == "1.000 s"


def test_formatter_json():
    """Test JSON formatting."""
    formatter = OutputFormatter(OutputFormat.JSON)
    data = [{"id": 1, "name": "test"}]
    output = formatter.format_output(data, title="Test")
    parsed = json.loads(output)
    assert parsed["title"] == "Test"
    assert parsed["data"] == data


def test_formatter_csv():
    """Test CSV formatting."""
    formatter = OutputFormatter(OutputFormat.CSV)
    data = [{"id": 1, "name": "test"}]
    output = formatter.format_output(data)
    lines = output.strip().split("\n")
    assert "id" in lines[0]
    assert "name" in lines[0]
    assert "1" in lines[1]
    assert "test" in lines[1]


def test_formatter_table():
    """Test table formatting."""
    formatter = OutputFormatter(OutputFormat.TABLE)
    data = [{"id": 1, "name": "test"}]
    output = formatter.format_output(data, headers=["id", "name"])
    assert "id" in output
    assert "name" in output
    assert "1" in output
    assert "test" in output


def test_formatter_table_single_dict():
    """Test table formatting with single dict."""
    formatter = OutputFormatter(OutputFormat.TABLE)
    data = {"id": 1, "name": "test"}
    output = formatter.format_output(data)
    assert "Field" in output
    assert "Value" in output
    assert "id" in output
    assert "test" in output


def test_formatter_format_value():
    """Test value formatting."""
    formatter = OutputFormatter(OutputFormat.TABLE)
    assert formatter._format_value(None) == "-"
    assert formatter._format_value(1000000) == "1,000,000"
    assert "1.50" in formatter._format_value(1.5)
    assert formatter._format_value([1, 2, 3]) == "1, 2, 3"
