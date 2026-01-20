"""Data models for TTNN profiling data."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BufferType(Enum):
    """Type of memory buffer."""

    L1 = "L1"
    DRAM = "DRAM"
    L1_SMALL = "L1_SMALL"
    TRACE = "TRACE"
    SYSTEM_MEMORY = "SYSTEM_MEMORY"

    @classmethod
    def from_int(cls, value: int) -> "BufferType":
        """Convert integer buffer type to enum."""
        mapping = {
            0: cls.DRAM,
            1: cls.L1,
            2: cls.SYSTEM_MEMORY,
            3: cls.L1_SMALL,
            4: cls.TRACE,
        }
        return mapping.get(value, cls.L1)


class TensorLayout(Enum):
    """Tensor data layout."""

    ROW_MAJOR = "ROW_MAJOR"
    TILE = "TILE"

    @classmethod
    def from_int(cls, value: int) -> "TensorLayout":
        """Convert integer layout to enum."""
        mapping = {
            0: cls.ROW_MAJOR,
            1: cls.TILE,
        }
        return mapping.get(value, cls.ROW_MAJOR)


class MemoryLayout(Enum):
    """Memory layout type."""

    INTERLEAVED = "INTERLEAVED"
    SINGLE_BANK = "SINGLE_BANK"
    HEIGHT_SHARDED = "HEIGHT_SHARDED"
    WIDTH_SHARDED = "WIDTH_SHARDED"
    BLOCK_SHARDED = "BLOCK_SHARDED"

    @classmethod
    def from_int(cls, value: int) -> "MemoryLayout":
        """Convert integer memory layout to enum."""
        mapping = {
            0: cls.INTERLEAVED,
            1: cls.SINGLE_BANK,
            2: cls.HEIGHT_SHARDED,
            3: cls.WIDTH_SHARDED,
            4: cls.BLOCK_SHARDED,
        }
        return mapping.get(value, cls.INTERLEAVED)


@dataclass
class Device:
    """Device information."""

    id: int
    num_y_cores: int
    num_x_cores: int
    num_y_compute_cores: int
    num_x_compute_cores: int
    worker_l1_size: int
    l1_num_banks: int
    l1_bank_size: int
    address_at_first_l1_bank: int
    address_at_first_l1_cb_buffer: int
    num_banks_per_storage_core: int
    num_compute_cores: int
    num_storage_cores: int
    total_l1_memory: int
    total_l1_for_tensors: int
    cb_limit: int
    arch: str = ""
    chip_id: int = 0

    @property
    def total_cores(self) -> int:
        """Total number of cores."""
        return self.num_y_cores * self.num_x_cores

    @property
    def total_compute_cores(self) -> int:
        """Total number of compute cores."""
        return self.num_y_compute_cores * self.num_x_compute_cores

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": self.id,
            "arch": self.arch,
            "chip_id": self.chip_id,
            "num_y_cores": self.num_y_cores,
            "num_x_cores": self.num_x_cores,
            "total_cores": self.total_cores,
            "num_y_compute_cores": self.num_y_compute_cores,
            "num_x_compute_cores": self.num_x_compute_cores,
            "total_compute_cores": self.total_compute_cores,
            "num_compute_cores": self.num_compute_cores,
            "num_storage_cores": self.num_storage_cores,
            "worker_l1_size": self.worker_l1_size,
            "l1_num_banks": self.l1_num_banks,
            "l1_bank_size": self.l1_bank_size,
            "total_l1_memory": self.total_l1_memory,
            "total_l1_for_tensors": self.total_l1_for_tensors,
            "cb_limit": self.cb_limit,
        }


@dataclass
class Operation:
    """Operation information."""

    id: int
    name: str
    duration: Optional[float] = None
    device_id: Optional[int] = None
    stack_trace_id: Optional[int] = None
    captured_graph_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "device_id": self.device_id,
            "stack_trace_id": self.stack_trace_id,
            "captured_graph_id": self.captured_graph_id,
        }


@dataclass
class OperationArgument:
    """Operation argument."""

    operation_id: int
    name: str
    value: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "value": self.value,
        }


@dataclass
class Tensor:
    """Tensor information."""

    id: int
    shape: str
    dtype: str
    layout: str
    memory_config: Optional[str] = None
    device_id: Optional[int] = None
    address: Optional[int] = None
    buffer_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": self.id,
            "shape": self.shape,
            "dtype": self.dtype,
            "layout": self.layout,
            "memory_config": self.memory_config,
            "device_id": self.device_id,
            "address": self.address,
            "buffer_type": self.buffer_type,
        }


@dataclass
class Buffer:
    """Buffer information."""

    id: int
    address: int
    max_size: int
    buffer_type: BufferType
    device_id: int
    operation_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": self.id,
            "address": self.address,
            "max_size": self.max_size,
            "buffer_type": self.buffer_type.value,
            "device_id": self.device_id,
            "operation_id": self.operation_id,
        }


@dataclass
class BufferPage:
    """Buffer page information (per-core memory allocation)."""

    buffer_id: int
    core_x: int
    core_y: int
    page_index: int
    page_address: int
    page_size: int
    device_id: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "buffer_id": self.buffer_id,
            "core_x": self.core_x,
            "core_y": self.core_y,
            "page_index": self.page_index,
            "page_address": self.page_address,
            "page_size": self.page_size,
            "device_id": self.device_id,
        }


@dataclass
class OperationPerf:
    """Operation performance data from CSV."""

    op_code: str
    op_name: str
    device_id: int
    core_count: int
    parallelization_strategy: str
    execution_time_ns: float
    host_time_ns: float
    math_utilization: float
    dram_read_bw: float
    dram_write_bw: float
    l1_read_bw: float
    l1_write_bw: float
    input_shapes: str = ""
    output_shapes: str = ""
    pm_ideal_ns: Optional[float] = None
    pm_compute_ns: Optional[float] = None
    pm_bandwidth_ns: Optional[float] = None
    operation_id: Optional[int] = None
    # New fields for performance report
    global_call_count: Optional[int] = None
    op_to_op_gap_ns: float = 0.0
    buffer_type: str = ""
    layout: str = ""
    math_fidelity: str = ""
    dram_bw_util_percent: float = 0.0
    fpu_util_percent: float = 0.0
    pm_req_i_bw: Optional[float] = None
    pm_req_o_bw: Optional[float] = None
    # Multi-CQ analysis fields
    dispatch_cq_cmd_time_ns: float = 0.0
    dispatch_wait_time_ns: float = 0.0
    erisc_kernel_duration_ns: float = 0.0

    @property
    def bound(self) -> str:
        """Determine if operation is compute or memory bound."""
        if self.pm_compute_ns is None or self.pm_bandwidth_ns is None:
            return ""
        if self.pm_compute_ns == 0 and self.pm_bandwidth_ns == 0:
            return ""
        if self.pm_compute_ns > self.pm_bandwidth_ns:
            return "Compute"
        elif self.pm_bandwidth_ns > self.pm_compute_ns:
            return "Memory"
        return "Balanced"

    @property
    def dram_bandwidth(self) -> Optional[float]:
        """Get total DRAM bandwidth (input + output)."""
        if self.pm_req_i_bw is not None and self.pm_req_o_bw is not None:
            return self.pm_req_i_bw + self.pm_req_o_bw
        return None

    @property
    def flops(self) -> Optional[float]:
        """Calculate FLOPs from compute metrics."""
        # FLOPs can be derived from pm_compute_ns if available
        return self.pm_compute_ns

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "op_code": self.op_code,
            "op_name": self.op_name,
            "device_id": self.device_id,
            "core_count": self.core_count,
            "parallelization_strategy": self.parallelization_strategy,
            "execution_time_ns": self.execution_time_ns,
            "host_time_ns": self.host_time_ns,
            "math_utilization": self.math_utilization,
            "dram_read_bw": self.dram_read_bw,
            "dram_write_bw": self.dram_write_bw,
            "l1_read_bw": self.l1_read_bw,
            "l1_write_bw": self.l1_write_bw,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "pm_ideal_ns": self.pm_ideal_ns,
            "pm_compute_ns": self.pm_compute_ns,
            "pm_bandwidth_ns": self.pm_bandwidth_ns,
            "operation_id": self.operation_id,
            "global_call_count": self.global_call_count,
            "op_to_op_gap_ns": self.op_to_op_gap_ns,
            "buffer_type": self.buffer_type,
            "layout": self.layout,
            "math_fidelity": self.math_fidelity,
            "dram_bw_util_percent": self.dram_bw_util_percent,
            "fpu_util_percent": self.fpu_util_percent,
            "bound": self.bound,
            "dispatch_cq_cmd_time_ns": self.dispatch_cq_cmd_time_ns,
            "dispatch_wait_time_ns": self.dispatch_wait_time_ns,
            "erisc_kernel_duration_ns": self.erisc_kernel_duration_ns,
        }


@dataclass
class MemorySummary:
    """Memory usage summary."""

    l1_used: int = 0
    l1_total: int = 0
    dram_used: int = 0
    dram_total: int = 0
    l1_buffer_count: int = 0
    dram_buffer_count: int = 0

    @property
    def l1_usage_percent(self) -> float:
        """L1 memory usage percentage."""
        if self.l1_total == 0:
            return 0.0
        return (self.l1_used / self.l1_total) * 100

    @property
    def dram_usage_percent(self) -> float:
        """DRAM memory usage percentage."""
        if self.dram_total == 0:
            return 0.0
        return (self.dram_used / self.dram_total) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "l1_used": self.l1_used,
            "l1_total": self.l1_total,
            "l1_usage_percent": round(self.l1_usage_percent, 2),
            "l1_buffer_count": self.l1_buffer_count,
            "dram_used": self.dram_used,
            "dram_total": self.dram_total,
            "dram_usage_percent": round(self.dram_usage_percent, 2),
            "dram_buffer_count": self.dram_buffer_count,
        }


@dataclass
class ReportInfo:
    """Report summary information."""

    profiler_path: Optional[str] = None
    performance_path: Optional[str] = None
    operation_count: int = 0
    tensor_count: int = 0
    buffer_count: int = 0
    device_count: int = 0
    total_duration_ns: float = 0.0
    devices: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "profiler_path": self.profiler_path,
            "performance_path": self.performance_path,
            "operation_count": self.operation_count,
            "tensor_count": self.tensor_count,
            "buffer_count": self.buffer_count,
            "device_count": self.device_count,
            "total_duration_ns": self.total_duration_ns,
            "total_duration_ms": round(self.total_duration_ns / 1_000_000, 3),
            "devices": [d.to_dict() for d in self.devices] if self.devices else [],
        }


@dataclass
class L1MemoryEntry:
    """L1 memory allocation entry with tensor details."""

    address: int
    size: int
    tensor_id: Optional[int] = None
    tensor_name: str = ""
    shape: str = ""
    dtype: str = ""
    memory_layout: str = ""
    buffer_type: str = "L1"
    operation_id: Optional[int] = None
    is_new: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "address": self.address,
            "size": self.size,
            "tensor_id": self.tensor_id,
            "tensor_name": self.tensor_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "memory_layout": self.memory_layout,
            "buffer_type": self.buffer_type,
            "operation_id": self.operation_id,
            "is_new": self.is_new,
        }
