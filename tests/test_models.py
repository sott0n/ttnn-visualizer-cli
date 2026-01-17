"""Tests for data models."""

from ttnn_vis_cli.data.models import (
    Buffer,
    BufferType,
    Device,
    MemorySummary,
    Operation,
    OperationPerf,
    ReportInfo,
    Tensor,
)


def test_buffer_type_from_int():
    """Test BufferType.from_int conversion."""
    assert BufferType.from_int(0) == BufferType.DRAM
    assert BufferType.from_int(1) == BufferType.L1
    assert BufferType.from_int(2) == BufferType.SYSTEM_MEMORY
    assert BufferType.from_int(3) == BufferType.L1_SMALL
    assert BufferType.from_int(4) == BufferType.TRACE
    assert BufferType.from_int(99) == BufferType.L1  # Default


def test_device_to_dict():
    """Test Device.to_dict method."""
    device = Device(
        id=0,
        num_y_cores=10,
        num_x_cores=12,
        num_y_compute_cores=8,
        num_x_compute_cores=8,
        worker_l1_size=1499136,
        l1_num_banks=64,
        l1_bank_size=1499136,
        address_at_first_l1_bank=0,
        address_at_first_l1_cb_buffer=0,
        num_banks_per_storage_core=4,
        num_compute_cores=64,
        num_storage_cores=16,
        total_l1_memory=95944704,
        total_l1_for_tensors=95944704,
        cb_limit=1499136,
    )
    d = device.to_dict()
    assert d["id"] == 0
    assert d["total_cores"] == 120
    assert d["total_compute_cores"] == 64


def test_operation_to_dict():
    """Test Operation.to_dict method."""
    op = Operation(id=1, name="ttnn.matmul", duration=1000000, device_id=0)
    d = op.to_dict()
    assert d["id"] == 1
    assert d["name"] == "ttnn.matmul"
    assert d["duration"] == 1000000


def test_tensor_to_dict():
    """Test Tensor.to_dict method."""
    tensor = Tensor(
        id=1,
        shape="[1, 32, 128, 128]",
        dtype="bfloat16",
        layout="TILE",
        memory_config="INTERLEAVED",
    )
    d = tensor.to_dict()
    assert d["id"] == 1
    assert d["shape"] == "[1, 32, 128, 128]"
    assert d["dtype"] == "bfloat16"


def test_buffer_to_dict():
    """Test Buffer.to_dict method."""
    buffer = Buffer(
        id=1,
        address=0,
        max_size=131072,
        buffer_type=BufferType.L1,
        device_id=0,
    )
    d = buffer.to_dict()
    assert d["id"] == 1
    assert d["buffer_type"] == "L1"
    assert d["max_size"] == 131072


def test_memory_summary():
    """Test MemorySummary properties."""
    summary = MemorySummary(
        l1_used=50000000,
        l1_total=100000000,
        dram_used=500000000,
        dram_total=1000000000,
        l1_buffer_count=10,
        dram_buffer_count=5,
    )
    assert summary.l1_usage_percent == 50.0
    assert summary.dram_usage_percent == 50.0

    d = summary.to_dict()
    assert d["l1_usage_percent"] == 50.0
    assert d["dram_usage_percent"] == 50.0


def test_operation_perf_to_dict():
    """Test OperationPerf.to_dict method."""
    perf = OperationPerf(
        op_code="MatmulDeviceOperation",
        op_name="ttnn.matmul",
        device_id=0,
        core_count=64,
        parallelization_strategy="SINGLE_CORE",
        execution_time_ns=1000000,
        host_time_ns=1500000,
        math_utilization=75.5,
        dram_read_bw=45.2,
        dram_write_bw=30.1,
        l1_read_bw=80.0,
        l1_write_bw=75.0,
    )
    d = perf.to_dict()
    assert d["op_code"] == "MatmulDeviceOperation"
    assert d["execution_time_ns"] == 1000000
    assert d["math_utilization"] == 75.5


def test_report_info_to_dict():
    """Test ReportInfo.to_dict method."""
    info = ReportInfo(
        profiler_path="/path/to/db.sqlite",
        operation_count=100,
        tensor_count=50,
        buffer_count=25,
        device_count=1,
        total_duration_ns=5000000000,
    )
    d = info.to_dict()
    assert d["operation_count"] == 100
    assert d["total_duration_ms"] == 5000.0
