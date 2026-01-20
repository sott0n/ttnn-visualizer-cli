"""Performance analysis logic for TTNN profiling data."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .models import OperationPerf


@dataclass
class OpDistribution:
    """Operation type distribution data."""

    op_code: str
    count: int
    total_time_ns: float
    avg_time_ns: float
    percent_time: float
    percent_count: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "op_code": self.op_code,
            "count": self.count,
            "total_time_ns": self.total_time_ns,
            "avg_time_ns": self.avg_time_ns,
            "percent_time": self.percent_time,
            "percent_count": self.percent_count,
        }


@dataclass
class CoreEfficiency:
    """Core efficiency analysis data."""

    core_count: int
    op_count: int
    total_time_ns: float
    avg_time_ns: float
    avg_fpu_util: float
    compute_bound: int
    memory_bound: int
    balanced: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "core_count": self.core_count,
            "op_count": self.op_count,
            "total_time_ns": self.total_time_ns,
            "avg_time_ns": self.avg_time_ns,
            "avg_fpu_util": self.avg_fpu_util,
            "compute_bound": self.compute_bound,
            "memory_bound": self.memory_bound,
            "balanced": self.balanced,
        }


@dataclass
class MatmulAnalysis:
    """Matmul/Conv operation analysis data."""

    global_call_count: Optional[int]
    core_count: int
    device_time_ns: float
    ideal_time_ns: Optional[float]
    efficiency: Optional[float]
    fpu_util: float
    bound: str
    math_fidelity: str
    op_code: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "global_call_count": self.global_call_count,
            "core_count": self.core_count,
            "device_time_ns": self.device_time_ns,
            "ideal_time_ns": self.ideal_time_ns,
            "efficiency": self.efficiency,
            "fpu_util": self.fpu_util,
            "bound": self.bound,
            "math_fidelity": self.math_fidelity,
            "op_code": self.op_code,
        }


@dataclass
class BottleneckInfo:
    """Bottleneck information."""

    global_call_count: Optional[int]
    op_code: str
    device_time_ns: float
    efficiency: Optional[float]
    issue: str
    category: str  # 'low_efficiency', 'high_gap', 'memory_inefficient'

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "global_call_count": self.global_call_count,
            "op_code": self.op_code,
            "device_time_ns": self.device_time_ns,
            "efficiency": self.efficiency,
            "issue": self.issue,
            "category": self.category,
        }


@dataclass
class AnalysisSummary:
    """Overall analysis summary."""

    total_operations: int
    total_device_time_ns: float
    total_op_to_op_gap_ns: float
    compute_bound_count: int
    memory_bound_count: int
    balanced_count: int
    avg_fpu_util: float
    avg_dram_util: float
    top_op_codes: list[tuple[str, int, float, float]]  # (op_code, count, time_ns, percent)
    low_efficiency_count: int
    high_gap_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_operations": self.total_operations,
            "total_device_time_ns": self.total_device_time_ns,
            "total_op_to_op_gap_ns": self.total_op_to_op_gap_ns,
            "compute_bound_count": self.compute_bound_count,
            "memory_bound_count": self.memory_bound_count,
            "balanced_count": self.balanced_count,
            "avg_fpu_util": self.avg_fpu_util,
            "avg_dram_util": self.avg_dram_util,
            "top_op_codes": [
                {"op_code": op, "count": cnt, "time_ns": time, "percent": pct}
                for op, cnt, time, pct in self.top_op_codes
            ],
            "low_efficiency_count": self.low_efficiency_count,
            "high_gap_count": self.high_gap_count,
        }


class PerfAnalyzer:
    """Performance analyzer for TTNN operations."""

    def __init__(self, operations: list[OperationPerf]):
        """Initialize with operation performance data.

        Args:
            operations: List of OperationPerf objects to analyze.
        """
        # Filter out signpost rows
        self.operations = [op for op in operations if op.op_name != "signpost"]
        self._total_time = sum(op.execution_time_ns for op in self.operations)

    def get_op_distribution(self, limit: int = 20) -> list[OpDistribution]:
        """Get operation type distribution.

        Args:
            limit: Maximum number of op codes to return.

        Returns:
            List of OpDistribution sorted by total time descending.
        """
        if not self.operations:
            return []

        # Group by op_code
        op_groups: dict[str, list[OperationPerf]] = defaultdict(list)
        for op in self.operations:
            op_code = op.op_code or "Unknown"
            op_groups[op_code].append(op)

        total_count = len(self.operations)
        distributions = []

        for op_code, ops in op_groups.items():
            total_time = sum(op.execution_time_ns for op in ops)
            count = len(ops)
            avg_time = total_time / count if count > 0 else 0

            distributions.append(
                OpDistribution(
                    op_code=op_code,
                    count=count,
                    total_time_ns=total_time,
                    avg_time_ns=avg_time,
                    percent_time=(total_time / self._total_time * 100)
                    if self._total_time > 0
                    else 0,
                    percent_count=(count / total_count * 100) if total_count > 0 else 0,
                )
            )

        # Sort by total time descending
        distributions.sort(key=lambda x: x.total_time_ns, reverse=True)

        return distributions[:limit]

    def get_core_efficiency(self) -> list[CoreEfficiency]:
        """Get core count efficiency analysis.

        Returns:
            List of CoreEfficiency sorted by core count.
        """
        if not self.operations:
            return []

        # Group by core count
        core_groups: dict[int, list[OperationPerf]] = defaultdict(list)
        for op in self.operations:
            if op.core_count and op.core_count > 0:
                core_groups[op.core_count].append(op)

        efficiencies = []

        for core_count, ops in core_groups.items():
            total_time = sum(op.execution_time_ns for op in ops)
            op_count = len(ops)
            avg_time = total_time / op_count if op_count > 0 else 0

            # Calculate average FPU utilization
            fpu_utils = [op.fpu_util_percent for op in ops if op.fpu_util_percent > 0]
            avg_fpu = sum(fpu_utils) / len(fpu_utils) if fpu_utils else 0

            # Count bound types
            compute_bound = sum(1 for op in ops if op.bound == "Compute")
            memory_bound = sum(1 for op in ops if op.bound == "Memory")
            balanced = sum(1 for op in ops if op.bound == "Balanced")

            efficiencies.append(
                CoreEfficiency(
                    core_count=core_count,
                    op_count=op_count,
                    total_time_ns=total_time,
                    avg_time_ns=avg_time,
                    avg_fpu_util=avg_fpu,
                    compute_bound=compute_bound,
                    memory_bound=memory_bound,
                    balanced=balanced,
                )
            )

        # Sort by core count
        efficiencies.sort(key=lambda x: x.core_count)

        return efficiencies

    def get_matmul_analysis(self, limit: int = 20) -> dict:
        """Get Matmul operations analysis.

        Args:
            limit: Maximum number of operations to return in detail.

        Returns:
            Dictionary with operations, summary, efficiency distribution, and fidelity breakdown.
        """
        return self._get_op_type_analysis(["Matmul", "MatmulDeviceOperation"], limit)

    def get_conv_analysis(self, limit: int = 20) -> dict:
        """Get Conv operations analysis.

        Args:
            limit: Maximum number of operations to return in detail.

        Returns:
            Dictionary with operations, summary, efficiency distribution, and fidelity breakdown.
        """
        return self._get_op_type_analysis(
            ["Conv", "Conv2d", "ConvDeviceOperation", "OptimizedConvNew"], limit
        )

    def _get_op_type_analysis(self, op_codes: list[str], limit: int) -> dict:
        """Get analysis for specific operation types.

        Args:
            op_codes: List of operation codes to analyze.
            limit: Maximum number of operations to return.

        Returns:
            Dictionary with operations, summary, efficiency distribution, and fidelity breakdown.
        """
        # Filter operations
        ops = [
            op
            for op in self.operations
            if any(code.lower() in (op.op_code or "").lower() for code in op_codes)
        ]

        if not ops:
            return {
                "operations": [],
                "summary": {
                    "total_count": 0,
                    "total_time_ns": 0,
                    "percent_of_all_ops": 0,
                    "avg_efficiency": 0,
                    "avg_fpu_util": 0,
                },
                "efficiency_distribution": {"high": 0, "medium": 0, "low": 0},
                "math_fidelity": {},
            }

        # Create operation analysis
        analyses = []
        for op in ops:
            efficiency = None
            if op.pm_ideal_ns and op.execution_time_ns > 0:
                efficiency = (op.pm_ideal_ns / op.execution_time_ns) * 100

            analyses.append(
                MatmulAnalysis(
                    global_call_count=op.global_call_count,
                    core_count=op.core_count,
                    device_time_ns=op.execution_time_ns,
                    ideal_time_ns=op.pm_ideal_ns,
                    efficiency=efficiency,
                    fpu_util=op.fpu_util_percent,
                    bound=op.bound,
                    math_fidelity=op.math_fidelity or "-",
                    op_code=op.op_code or "",
                )
            )

        # Sort by device time descending
        analyses.sort(key=lambda x: x.device_time_ns, reverse=True)

        # Calculate summary
        total_time = sum(op.device_time_ns for op in analyses)
        efficiencies = [op.efficiency for op in analyses if op.efficiency is not None]
        fpu_utils = [op.fpu_util for op in analyses if op.fpu_util > 0]

        # Efficiency distribution
        high_eff = sum(1 for e in efficiencies if e and e > 80)
        med_eff = sum(1 for e in efficiencies if e and 50 <= e <= 80)
        low_eff = sum(1 for e in efficiencies if e and e < 50)

        # Math fidelity breakdown
        fidelity_counts: dict[str, int] = defaultdict(int)
        for op in analyses:
            if op.math_fidelity and op.math_fidelity != "-":
                fidelity_counts[op.math_fidelity] += 1

        return {
            "operations": [op.to_dict() for op in analyses[:limit]],
            "summary": {
                "total_count": len(analyses),
                "total_time_ns": total_time,
                "percent_of_all_ops": (total_time / self._total_time * 100)
                if self._total_time > 0
                else 0,
                "avg_efficiency": sum(efficiencies) / len(efficiencies)
                if efficiencies
                else 0,
                "avg_fpu_util": sum(fpu_utils) / len(fpu_utils) if fpu_utils else 0,
            },
            "efficiency_distribution": {
                "high": high_eff,
                "medium": med_eff,
                "low": low_eff,
            },
            "math_fidelity": dict(fidelity_counts),
        }

    def get_bottlenecks(
        self,
        efficiency_threshold: float = 50.0,
        gap_threshold_ms: float = 100.0,
    ) -> dict:
        """Identify performance bottlenecks.

        Args:
            efficiency_threshold: FPU utilization threshold for low efficiency.
            gap_threshold_ms: Op-to-op gap threshold in milliseconds.

        Returns:
            Dictionary with categorized bottleneck lists.
        """
        low_efficiency = []
        high_gap = []
        memory_inefficient = []

        gap_threshold_ns = gap_threshold_ms * 1_000_000

        for op in self.operations:
            # Low efficiency operations
            if op.fpu_util_percent > 0 and op.fpu_util_percent < efficiency_threshold:
                low_efficiency.append(
                    BottleneckInfo(
                        global_call_count=op.global_call_count,
                        op_code=op.op_code or "Unknown",
                        device_time_ns=op.execution_time_ns,
                        efficiency=op.fpu_util_percent,
                        issue=f"Low FPU utilization ({op.fpu_util_percent:.1f}%)",
                        category="low_efficiency",
                    )
                )

            # High op-to-op gap
            if op.op_to_op_gap_ns > gap_threshold_ns:
                high_gap.append(
                    BottleneckInfo(
                        global_call_count=op.global_call_count,
                        op_code=op.op_code or "Unknown",
                        device_time_ns=op.op_to_op_gap_ns,
                        efficiency=None,
                        issue="Host overhead / data transfer",
                        category="high_gap",
                    )
                )

            # Memory-bound with low DRAM utilization
            if (
                op.bound == "Memory"
                and op.dram_bw_util_percent > 0
                and op.dram_bw_util_percent < 50
            ):
                memory_inefficient.append(
                    BottleneckInfo(
                        global_call_count=op.global_call_count,
                        op_code=op.op_code or "Unknown",
                        device_time_ns=op.execution_time_ns,
                        efficiency=op.dram_bw_util_percent,
                        issue=f"Memory-bound with low DRAM utilization ({op.dram_bw_util_percent:.1f}%)",
                        category="memory_inefficient",
                    )
                )

        # Sort by device time descending
        low_efficiency.sort(key=lambda x: x.device_time_ns, reverse=True)
        high_gap.sort(key=lambda x: x.device_time_ns, reverse=True)
        memory_inefficient.sort(key=lambda x: x.device_time_ns, reverse=True)

        return {
            "low_efficiency": [b.to_dict() for b in low_efficiency[:20]],
            "high_gap": [b.to_dict() for b in high_gap[:20]],
            "memory_inefficient": [b.to_dict() for b in memory_inefficient[:20]],
            "summary": {
                "low_efficiency_count": len(low_efficiency),
                "high_gap_count": len(high_gap),
                "memory_inefficient_count": len(memory_inefficient),
            },
        }

    def get_summary(self) -> AnalysisSummary:
        """Get overall analysis summary.

        Returns:
            AnalysisSummary with all key metrics.
        """
        if not self.operations:
            return AnalysisSummary(
                total_operations=0,
                total_device_time_ns=0,
                total_op_to_op_gap_ns=0,
                compute_bound_count=0,
                memory_bound_count=0,
                balanced_count=0,
                avg_fpu_util=0,
                avg_dram_util=0,
                top_op_codes=[],
                low_efficiency_count=0,
                high_gap_count=0,
            )

        # Basic counts
        total_ops = len(self.operations)
        total_time = self._total_time
        total_gap = sum(op.op_to_op_gap_ns for op in self.operations)

        # Bound distribution
        compute_bound = sum(1 for op in self.operations if op.bound == "Compute")
        memory_bound = sum(1 for op in self.operations if op.bound == "Memory")
        balanced = sum(1 for op in self.operations if op.bound == "Balanced")

        # Utilization
        fpu_utils = [
            op.fpu_util_percent for op in self.operations if op.fpu_util_percent > 0
        ]
        dram_utils = [
            op.dram_bw_util_percent
            for op in self.operations
            if op.dram_bw_util_percent > 0
        ]
        avg_fpu = sum(fpu_utils) / len(fpu_utils) if fpu_utils else 0
        avg_dram = sum(dram_utils) / len(dram_utils) if dram_utils else 0

        # Top op codes
        op_distribution = self.get_op_distribution(limit=5)
        top_op_codes = [
            (d.op_code, d.count, d.total_time_ns, d.percent_time)
            for d in op_distribution
        ]

        # Bottleneck counts
        bottlenecks = self.get_bottlenecks()
        low_eff_count = bottlenecks["summary"]["low_efficiency_count"]
        high_gap_count = bottlenecks["summary"]["high_gap_count"]

        return AnalysisSummary(
            total_operations=total_ops,
            total_device_time_ns=total_time,
            total_op_to_op_gap_ns=total_gap,
            compute_bound_count=compute_bound,
            memory_bound_count=memory_bound,
            balanced_count=balanced,
            avg_fpu_util=avg_fpu,
            avg_dram_util=avg_dram,
            top_op_codes=top_op_codes,
            low_efficiency_count=low_eff_count,
            high_gap_count=high_gap_count,
        )
