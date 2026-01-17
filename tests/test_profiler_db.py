"""Tests for ProfilerDB."""

import pytest

from ttnn_vis_cli.data.models import BufferType
from ttnn_vis_cli.data.profiler_db import ProfilerDB


def test_profiler_db_init(temp_db):
    """Test ProfilerDB initialization."""
    db = ProfilerDB(temp_db)
    assert db.db_path.exists()


def test_profiler_db_file_not_found():
    """Test ProfilerDB raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        ProfilerDB("/nonexistent/path/db.sqlite")


def test_get_devices(temp_db):
    """Test getting devices."""
    db = ProfilerDB(temp_db)
    devices = db.get_devices()
    assert len(devices) == 1
    assert devices[0].id == 0
    assert devices[0].num_compute_cores == 64


def test_get_device(temp_db):
    """Test getting a specific device."""
    db = ProfilerDB(temp_db)
    device = db.get_device(0)
    assert device is not None
    assert device.id == 0

    device = db.get_device(999)
    assert device is None


def test_get_operations(temp_db):
    """Test getting operations."""
    db = ProfilerDB(temp_db)
    ops = db.get_operations()
    assert len(ops) == 3
    assert ops[0].name == "ttnn.matmul"


def test_get_operations_top(temp_db):
    """Test getting top operations by duration."""
    db = ProfilerDB(temp_db)
    ops = db.get_operations(limit=2, order_by_duration=True)
    assert len(ops) == 2
    assert ops[0].name == "ttnn.matmul"  # Longest duration
    assert ops[1].name == "ttnn.add"


def test_get_operation(temp_db):
    """Test getting a specific operation."""
    db = ProfilerDB(temp_db)
    op = db.get_operation(1)
    assert op is not None
    assert op.name == "ttnn.matmul"
    assert op.duration == 1000000

    op = db.get_operation(999)
    assert op is None


def test_get_operation_arguments(temp_db):
    """Test getting operation arguments."""
    db = ProfilerDB(temp_db)
    args = db.get_operation_arguments(1)
    assert len(args) == 2
    arg_names = {arg.name for arg in args}
    assert "transpose_a" in arg_names
    assert "transpose_b" in arg_names


def test_get_tensors(temp_db):
    """Test getting tensors."""
    db = ProfilerDB(temp_db)
    tensors = db.get_tensors()
    assert len(tensors) == 3
    assert tensors[0].dtype == "bfloat16"


def test_get_tensor(temp_db):
    """Test getting a specific tensor."""
    db = ProfilerDB(temp_db)
    tensor = db.get_tensor(1)
    assert tensor is not None
    assert tensor.shape == "[1, 32, 128, 128]"

    tensor = db.get_tensor(999)
    assert tensor is None


def test_get_input_tensors(temp_db):
    """Test getting input tensors for an operation."""
    db = ProfilerDB(temp_db)
    tensors = db.get_input_tensors(1)
    assert len(tensors) == 2


def test_get_output_tensors(temp_db):
    """Test getting output tensors for an operation."""
    db = ProfilerDB(temp_db)
    tensors = db.get_output_tensors(1)
    assert len(tensors) == 1
    assert tensors[0].id == 3


def test_get_buffers(temp_db):
    """Test getting buffers."""
    db = ProfilerDB(temp_db)
    buffers = db.get_buffers()
    assert len(buffers) == 3


def test_get_buffers_by_type(temp_db):
    """Test filtering buffers by type."""
    db = ProfilerDB(temp_db)
    l1_buffers = db.get_buffers(buffer_type=BufferType.L1)
    assert len(l1_buffers) == 2
    assert all(b.buffer_type == BufferType.L1 for b in l1_buffers)

    dram_buffers = db.get_buffers(buffer_type=BufferType.DRAM)
    assert len(dram_buffers) == 1


def test_get_buffers_by_operation(temp_db):
    """Test filtering buffers by operation."""
    db = ProfilerDB(temp_db)
    buffers = db.get_buffers(operation_id=1)
    assert len(buffers) == 2


def test_get_memory_summary(temp_db):
    """Test getting memory summary."""
    db = ProfilerDB(temp_db)
    summary = db.get_memory_summary()
    assert summary.l1_buffer_count == 2
    assert summary.dram_buffer_count == 1
    assert summary.l1_used == 131072 + 65536
    assert summary.dram_used == 1048576


def test_get_stack_trace(temp_db):
    """Test getting stack trace."""
    db = ProfilerDB(temp_db)
    trace = db.get_stack_trace(1)
    assert trace is not None
    assert "test.py" in trace

    trace = db.get_stack_trace(999)
    assert trace is None


def test_get_report_info(temp_db):
    """Test getting report info."""
    db = ProfilerDB(temp_db)
    info = db.get_report_info()
    assert info.operation_count == 3
    assert info.tensor_count == 3
    assert info.buffer_count == 3
    assert info.device_count == 1


def test_get_table_names(temp_db):
    """Test getting table names."""
    db = ProfilerDB(temp_db)
    tables = db.get_table_names()
    assert "devices" in tables
    assert "operations" in tables
    assert "tensors" in tables
    assert "buffers" in tables
