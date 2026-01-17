"""Tests for PerfCSV."""

from pathlib import Path

import pytest

from ttnn_vis_cli.data.perf_csv import PerfCSV


def test_perf_csv_init(temp_perf_csv):
    """Test PerfCSV initialization with CSV file."""
    perf = PerfCSV(temp_perf_csv)
    assert perf.is_valid()
    assert perf.csv_file == Path(temp_perf_csv)


def test_perf_csv_init_dir(temp_perf_dir):
    """Test PerfCSV initialization with directory."""
    perf = PerfCSV(temp_perf_dir)
    assert perf.is_valid()


def test_perf_csv_invalid():
    """Test PerfCSV with non-existent path."""
    perf = PerfCSV("/nonexistent/path")
    assert not perf.is_valid()


def test_get_operations(temp_perf_csv):
    """Test getting operations."""
    perf = PerfCSV(temp_perf_csv)
    ops = perf.get_operations()
    assert len(ops) == 3
    assert ops[0].op_name == "ttnn.matmul"


def test_get_operations_by_time(temp_perf_csv):
    """Test getting operations ordered by time."""
    perf = PerfCSV(temp_perf_csv)
    ops = perf.get_operations(order_by_time=True)
    assert len(ops) == 3
    assert ops[0].execution_time_ns >= ops[1].execution_time_ns
    assert ops[1].execution_time_ns >= ops[2].execution_time_ns


def test_get_operations_limit(temp_perf_csv):
    """Test getting limited operations."""
    perf = PerfCSV(temp_perf_csv)
    ops = perf.get_operations(limit=2)
    assert len(ops) == 2


def test_get_top_operations(temp_perf_csv):
    """Test getting top operations."""
    perf = PerfCSV(temp_perf_csv)
    ops = perf.get_top_operations(2)
    assert len(ops) == 2
    assert ops[0].op_name == "ttnn.matmul"  # Longest execution time


def test_get_summary(temp_perf_csv):
    """Test getting summary."""
    perf = PerfCSV(temp_perf_csv)
    summary = perf.get_summary()
    assert summary["total_operations"] == 3
    assert summary["total_execution_time_ns"] == 1750000
    assert summary["max_execution_time_ns"] == 1000000
    assert summary["min_execution_time_ns"] == 250000


def test_get_raw_dataframe(temp_perf_csv):
    """Test getting raw dataframe."""
    perf = PerfCSV(temp_perf_csv)
    df = perf.get_raw_dataframe()
    assert df is not None
    assert len(df) == 3


def test_operation_values(temp_perf_csv):
    """Test that operation values are correctly parsed."""
    perf = PerfCSV(temp_perf_csv)
    ops = perf.get_operations()
    matmul = ops[0]
    assert matmul.op_code == "MatmulDeviceOperation"
    assert matmul.device_id == 0
    assert matmul.core_count == 64
    assert matmul.execution_time_ns == 1000000
    assert matmul.math_utilization == 75.5
