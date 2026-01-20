"""Performance CSV data access for TTNN profiling data."""

import csv
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import OperationPerf


class PerfCSV:
    """Access layer for TTNN performance CSV files."""

    def __init__(self, perf_path: str | Path):
        """Initialize with path to performance report directory.

        Args:
            perf_path: Path to the performance report directory or CSV file.
        """
        self.perf_path = Path(perf_path)
        self._csv_file: Optional[Path] = None
        self._find_csv_file()

    def _find_csv_file(self) -> None:
        """Find the ops_perf_results CSV file."""
        if self.perf_path.is_file() and self.perf_path.suffix == ".csv":
            self._csv_file = self.perf_path
        elif self.perf_path.is_dir():
            # Look for ops_perf_results*.csv files
            csv_files = list(self.perf_path.glob("ops_perf_results*.csv"))
            if csv_files:
                # Use the most recent one
                self._csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
            else:
                # Look in subdirectories
                csv_files = list(self.perf_path.rglob("ops_perf_results*.csv"))
                if csv_files:
                    self._csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)

    @property
    def csv_file(self) -> Optional[Path]:
        """Get the CSV file path."""
        return self._csv_file

    def is_valid(self) -> bool:
        """Check if performance data is available."""
        return self._csv_file is not None and self._csv_file.exists()

    def get_operations(
        self,
        limit: Optional[int] = None,
        order_by_time: bool = False,
    ) -> list[OperationPerf]:
        """Get operation performance data.

        Args:
            limit: Maximum number of operations to return.
            order_by_time: If True, order by execution time descending.

        Returns:
            List of OperationPerf objects.
        """
        if not self.is_valid():
            return []

        try:
            df = pd.read_csv(self._csv_file)
        except Exception:
            return []

        # Normalize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Map columns to standard names
        column_mapping = {
            "op_code": ["op_code", "opcode", "op"],
            "op_name": ["op_name", "name", "operation_name", "op_type"],
            "device_id": ["device_id", "device"],
            "core_count": ["core_count", "cores", "num_cores"],
            "parallelization_strategy": ["parallelization_strategy", "strategy"],
            "execution_time_ns": [
                "device_kernel_duration_[ns]",
                "device_kernel_duration_ns",
                "kernel_duration_[ns]",
                "kernel_duration_ns",
                "execution_time_ns",
                "exec_time_ns",
                "duration_ns",
            ],
            "host_time_ns": ["host_time_ns", "host_duration_ns", "host_duration_[ns]"],
            "math_utilization": [
                "math_utilization",
                "math_util",
                "compute_utilization",
            ],
            "dram_read_bw": [
                "output_dram_bw_peak_utilization_[%]",
                "dram_read_bw",
                "dram_bw_read",
            ],
            "dram_write_bw": [
                "dram_write_bw",
                "dram_bw_write",
            ],
            "l1_read_bw": ["l1_read_bw", "l1_bw_read"],
            "l1_write_bw": ["l1_write_bw", "l1_bw_write"],
            "input_shapes": ["input_shapes", "inputs"],
            "output_shapes": ["output_shapes", "outputs"],
            "pm_ideal_ns": ["pm_ideal_[ns]", "pm_ideal_ns"],
            "pm_compute_ns": ["pm_compute_[ns]", "pm_compute_ns"],
            "pm_bandwidth_ns": ["pm_bandwidth_[ns]", "pm_bandwidth_ns"],
            # New fields for performance report
            "global_call_count": ["global_call_count"],
            "op_to_op_gap_ns": ["op_to_op_latency_[ns]", "op_to_op_latency_ns"],
            "input_0_memory": ["input_0_memory"],
            "input_0_layout": ["input_0_layout"],
            "math_fidelity": ["math_fidelity"],
            "dram_bw_util_percent": ["dram_bw_util_(%)", "dram_bw_util_(%)"],
            "fpu_util_percent": ["pm_fpu_util_(%)", "pm_fpu_util_(%)"],
            "pm_req_i_bw": ["pm_req_i_bw"],
            "pm_req_o_bw": ["pm_req_o_bw"],
        }

        # Create normalized dataframe
        normalized_data = {}
        for target_col, source_cols in column_mapping.items():
            for source_col in source_cols:
                if source_col in df.columns:
                    normalized_data[target_col] = df[source_col]
                    break
            if target_col not in normalized_data:
                normalized_data[target_col] = None

        # Create operations list
        operations = []
        for idx in range(len(df)):
            try:
                # Extract buffer type from memory config string
                buffer_type = self._extract_buffer_type(
                    self._get_value(normalized_data, "input_0_memory", idx, "")
                )
                op = OperationPerf(
                    op_code=self._get_value(normalized_data, "op_code", idx, ""),
                    op_name=self._get_value(normalized_data, "op_name", idx, ""),
                    device_id=int(self._get_value(normalized_data, "device_id", idx, 0)),
                    core_count=int(self._get_value(normalized_data, "core_count", idx, 0)),
                    parallelization_strategy=self._get_value(
                        normalized_data, "parallelization_strategy", idx, ""
                    ),
                    execution_time_ns=float(
                        self._get_value(normalized_data, "execution_time_ns", idx, 0)
                    ),
                    host_time_ns=float(
                        self._get_value(normalized_data, "host_time_ns", idx, 0)
                    ),
                    math_utilization=float(
                        self._get_value(normalized_data, "math_utilization", idx, 0)
                    ),
                    dram_read_bw=float(
                        self._get_value(normalized_data, "dram_read_bw", idx, 0)
                    ),
                    dram_write_bw=float(
                        self._get_value(normalized_data, "dram_write_bw", idx, 0)
                    ),
                    l1_read_bw=float(
                        self._get_value(normalized_data, "l1_read_bw", idx, 0)
                    ),
                    l1_write_bw=float(
                        self._get_value(normalized_data, "l1_write_bw", idx, 0)
                    ),
                    input_shapes=self._get_value(normalized_data, "input_shapes", idx, ""),
                    output_shapes=self._get_value(normalized_data, "output_shapes", idx, ""),
                    pm_ideal_ns=self._get_optional_float(
                        normalized_data, "pm_ideal_ns", idx
                    ),
                    pm_compute_ns=self._get_optional_float(
                        normalized_data, "pm_compute_ns", idx
                    ),
                    pm_bandwidth_ns=self._get_optional_float(
                        normalized_data, "pm_bandwidth_ns", idx
                    ),
                    # New fields
                    global_call_count=self._get_optional_int(
                        normalized_data, "global_call_count", idx
                    ),
                    op_to_op_gap_ns=float(
                        self._get_value(normalized_data, "op_to_op_gap_ns", idx, 0)
                    ),
                    buffer_type=buffer_type,
                    layout=self._get_value(normalized_data, "input_0_layout", idx, ""),
                    math_fidelity=self._get_value(normalized_data, "math_fidelity", idx, ""),
                    dram_bw_util_percent=float(
                        self._get_value(normalized_data, "dram_bw_util_percent", idx, 0)
                    ),
                    fpu_util_percent=float(
                        self._get_value(normalized_data, "fpu_util_percent", idx, 0)
                    ),
                    pm_req_i_bw=self._get_optional_float(
                        normalized_data, "pm_req_i_bw", idx
                    ),
                    pm_req_o_bw=self._get_optional_float(
                        normalized_data, "pm_req_o_bw", idx
                    ),
                )
                operations.append(op)
            except (ValueError, KeyError):
                continue

        if order_by_time:
            operations.sort(key=lambda x: x.execution_time_ns, reverse=True)

        if limit:
            operations = operations[:limit]

        return operations

    def _get_value(
        self, data: dict, key: str, idx: int, default: str | int | float
    ) -> str | int | float:
        """Get a value from normalized data."""
        if data.get(key) is None:
            return default
        try:
            val = data[key].iloc[idx]
            if pd.isna(val):
                return default
            return val
        except (IndexError, AttributeError):
            return default

    def _get_optional_float(
        self, data: dict, key: str, idx: int
    ) -> Optional[float]:
        """Get an optional float value."""
        if data.get(key) is None:
            return None
        try:
            val = data[key].iloc[idx]
            if pd.isna(val):
                return None
            return float(val)
        except (IndexError, ValueError, AttributeError):
            return None

    def _get_optional_int(
        self, data: dict, key: str, idx: int
    ) -> Optional[int]:
        """Get an optional int value."""
        if data.get(key) is None:
            return None
        try:
            val = data[key].iloc[idx]
            if pd.isna(val):
                return None
            return int(val)
        except (IndexError, ValueError, AttributeError):
            return None

    def _extract_buffer_type(self, memory_str: str) -> str:
        """Extract buffer type from memory config string like 'DEV_1_L1_HEIGHT_SHARDED'."""
        if not memory_str:
            return ""
        memory_str = str(memory_str).upper()
        if "DRAM" in memory_str:
            return "DRAM"
        elif "L1" in memory_str:
            return "L1"
        elif "SYSTEM_MEMORY" in memory_str:
            return "System"
        return memory_str

    def get_summary(self) -> dict:
        """Get performance summary statistics."""
        operations = self.get_operations()
        if not operations:
            return {
                "total_operations": 0,
                "total_execution_time_ns": 0,
                "total_execution_time_ms": 0,
                "avg_execution_time_ns": 0,
                "max_execution_time_ns": 0,
                "min_execution_time_ns": 0,
            }

        exec_times = [op.execution_time_ns for op in operations if op.execution_time_ns > 0]
        total_time = sum(exec_times)

        return {
            "total_operations": len(operations),
            "total_execution_time_ns": total_time,
            "total_execution_time_ms": round(total_time / 1_000_000, 3),
            "avg_execution_time_ns": round(total_time / len(exec_times), 3) if exec_times else 0,
            "max_execution_time_ns": max(exec_times) if exec_times else 0,
            "min_execution_time_ns": min(exec_times) if exec_times else 0,
            "avg_math_utilization": round(
                sum(op.math_utilization for op in operations) / len(operations), 3
            ) if operations else 0,
            "csv_file": str(self._csv_file) if self._csv_file else None,
        }

    def get_top_operations(self, n: int = 10) -> list[OperationPerf]:
        """Get top N operations by execution time."""
        return self.get_operations(limit=n, order_by_time=True)

    def get_raw_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the raw pandas DataFrame."""
        if not self.is_valid():
            return None
        try:
            return pd.read_csv(self._csv_file)
        except Exception:
            return None
