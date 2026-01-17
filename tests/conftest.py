"""Pytest configuration and fixtures."""

import csv
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE devices (
            device_id INTEGER PRIMARY KEY,
            num_y_cores INTEGER,
            num_x_cores INTEGER,
            num_y_compute_cores INTEGER,
            num_x_compute_cores INTEGER,
            worker_l1_size INTEGER,
            l1_num_banks INTEGER,
            l1_bank_size INTEGER,
            address_at_first_l1_bank INTEGER,
            address_at_first_l1_cb_buffer INTEGER,
            num_banks_per_storage_core INTEGER,
            num_compute_cores INTEGER,
            num_storage_cores INTEGER,
            total_l1_memory INTEGER,
            total_l1_for_tensors INTEGER,
            cb_limit INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE operations (
            operation_id INTEGER PRIMARY KEY,
            name TEXT,
            duration REAL,
            device_id INTEGER,
            stack_trace_id INTEGER,
            captured_graph_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE operation_arguments (
            operation_id INTEGER,
            name TEXT,
            value TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE tensors (
            tensor_id INTEGER PRIMARY KEY,
            shape TEXT,
            dtype TEXT,
            layout TEXT,
            memory_config TEXT,
            device_id INTEGER,
            address INTEGER,
            buffer_type TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE input_tensors (
            operation_id INTEGER,
            tensor_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE output_tensors (
            operation_id INTEGER,
            tensor_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE buffers (
            buffer_id INTEGER PRIMARY KEY,
            address INTEGER,
            max_size INTEGER,
            buffer_type INTEGER,
            device_id INTEGER,
            operation_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE buffer_pages (
            buffer_id INTEGER,
            core_x INTEGER,
            core_y INTEGER,
            page_index INTEGER,
            page_address INTEGER,
            page_size INTEGER,
            device_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE stack_traces (
            stack_trace_id INTEGER PRIMARY KEY,
            stack_trace TEXT
        )
    """)

    # Insert test data
    cursor.execute("""
        INSERT INTO devices VALUES (
            0, 10, 12, 8, 8, 1499136, 64, 1499136, 0, 0, 4, 64, 16, 95944704, 95944704, 1499136
        )
    """)

    cursor.execute("""
        INSERT INTO operations VALUES (1, 'ttnn.matmul', 1000000, 0, 1, NULL)
    """)
    cursor.execute("""
        INSERT INTO operations VALUES (2, 'ttnn.add', 500000, 0, NULL, NULL)
    """)
    cursor.execute("""
        INSERT INTO operations VALUES (3, 'ttnn.relu', 250000, 0, NULL, NULL)
    """)

    cursor.execute("""
        INSERT INTO operation_arguments VALUES (1, 'transpose_a', 'False')
    """)
    cursor.execute("""
        INSERT INTO operation_arguments VALUES (1, 'transpose_b', 'True')
    """)

    cursor.execute("""
        INSERT INTO tensors VALUES (1, '[1, 32, 128, 128]', 'bfloat16', 'TILE', 'INTERLEAVED', 0, 0, 'L1')
    """)
    cursor.execute("""
        INSERT INTO tensors VALUES (2, '[1, 32, 128, 64]', 'bfloat16', 'TILE', 'INTERLEAVED', 0, 131072, 'L1')
    """)
    cursor.execute("""
        INSERT INTO tensors VALUES (3, '[1, 32, 128, 64]', 'bfloat16', 'TILE', 'INTERLEAVED', 0, 262144, 'L1')
    """)

    cursor.execute("""
        INSERT INTO input_tensors VALUES (1, 1)
    """)
    cursor.execute("""
        INSERT INTO input_tensors VALUES (1, 2)
    """)
    cursor.execute("""
        INSERT INTO output_tensors VALUES (1, 3)
    """)

    cursor.execute("""
        INSERT INTO buffers VALUES (1, 0, 131072, 1, 0, 1)
    """)
    cursor.execute("""
        INSERT INTO buffers VALUES (2, 131072, 65536, 1, 0, 1)
    """)
    cursor.execute("""
        INSERT INTO buffers VALUES (3, 1000000, 1048576, 0, 0, NULL)
    """)

    cursor.execute("""
        INSERT INTO stack_traces VALUES (1, 'File "test.py", line 10, in main\n    model(x)')
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_perf_csv():
    """Create a temporary performance CSV file with test data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="ops_perf_results_", delete=False
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "OP CODE", "OP NAME", "DEVICE ID", "CORE COUNT",
            "PARALLELIZATION STRATEGY", "DEVICE KERNEL DURATION [ns]",
            "HOST TIME NS", "MATH UTILIZATION",
            "OUTPUT DRAM BW PEAK UTILIZATION [%]", "DRAM WRITE BW",
            "L1 READ BW", "L1 WRITE BW"
        ])
        writer.writerow([
            "MatmulDeviceOperation", "ttnn.matmul", 0, 64,
            "SINGLE_CORE", 1000000, 1500000, 75.5, 45.2, 30.1, 80.0, 75.0
        ])
        writer.writerow([
            "AddAndApplyActivationToOutput", "ttnn.add", 0, 32,
            "SHARDED", 500000, 600000, 50.0, 20.0, 15.0, 60.0, 55.0
        ])
        writer.writerow([
            "UnaryDeviceOperation", "ttnn.relu", 0, 64,
            "INTERLEAVED", 250000, 300000, 30.0, 10.0, 8.0, 40.0, 35.0
        ])
        csv_path = f.name

    yield csv_path

    # Cleanup
    Path(csv_path).unlink(missing_ok=True)


@pytest.fixture
def temp_perf_dir(temp_perf_csv):
    """Create a temporary directory containing the performance CSV."""
    csv_path = Path(temp_perf_csv)
    return str(csv_path.parent)
