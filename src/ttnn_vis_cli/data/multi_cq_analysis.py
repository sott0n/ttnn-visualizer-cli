"""Multi-CQ analysis logic for TTNN profiling data."""

from dataclasses import dataclass

from .models import OperationPerf

# Thresholds for recommendations
IO_BOTTLENECK_THRESHOLD = 30  # I/O bottleneck if I/O time > 30% of total
MULTI_CQ_RECOMMENDED_THRESHOLD = 20  # Recommend 2CQ if I/O overhead > 20%
IO_DOMINANCE_THRESHOLD = 0.5  # Component dominates I/O if > 50% of total


@dataclass
class MultiCQSummary:
    """Summary of multi-CQ analysis."""

    total_operations: int
    total_device_time_ns: float
    total_io_time_ns: float  # dispatch + wait + erisc
    total_dispatch_cq_time_ns: float
    total_wait_time_ns: float
    total_erisc_time_ns: float
    total_compute_time_ns: float
    io_overhead_percent: float
    is_io_bound: bool
    multi_cq_recommended: bool
    io_bound_operations: int
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_operations": self.total_operations,
            "total_device_time_ns": self.total_device_time_ns,
            "total_device_time_ms": round(self.total_device_time_ns / 1_000_000, 3),
            "total_io_time_ns": self.total_io_time_ns,
            "total_io_time_ms": round(self.total_io_time_ns / 1_000_000, 3),
            "total_dispatch_cq_time_ns": self.total_dispatch_cq_time_ns,
            "total_dispatch_cq_time_ms": round(self.total_dispatch_cq_time_ns / 1_000_000, 3),
            "total_wait_time_ns": self.total_wait_time_ns,
            "total_wait_time_ms": round(self.total_wait_time_ns / 1_000_000, 3),
            "total_erisc_time_ns": self.total_erisc_time_ns,
            "total_erisc_time_ms": round(self.total_erisc_time_ns / 1_000_000, 3),
            "total_compute_time_ns": self.total_compute_time_ns,
            "total_compute_time_ms": round(self.total_compute_time_ns / 1_000_000, 3),
            "io_overhead_percent": round(self.io_overhead_percent, 2),
            "is_io_bound": self.is_io_bound,
            "multi_cq_recommended": self.multi_cq_recommended,
            "io_bound_operations": self.io_bound_operations,
            "recommendations": self.recommendations,
        }


@dataclass
class OperationIOAnalysis:
    """I/O analysis for a single operation."""

    op_code: str
    op_name: str
    device_time_ns: float
    dispatch_time_ns: float
    wait_time_ns: float
    erisc_time_ns: float
    io_overhead_percent: float
    is_io_bound: bool

    @property
    def total_io_time_ns(self) -> float:
        """Total I/O time (dispatch + wait + erisc)."""
        return self.dispatch_time_ns + self.wait_time_ns + self.erisc_time_ns

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "op_code": self.op_code,
            "op_name": self.op_name,
            "device_time_ns": self.device_time_ns,
            "device_time_us": round(self.device_time_ns / 1_000, 2),
            "total_io_time_ns": self.total_io_time_ns,
            "total_io_time_us": round(self.total_io_time_ns / 1_000, 2),
            "dispatch_time_ns": self.dispatch_time_ns,
            "dispatch_time_us": round(self.dispatch_time_ns / 1_000, 2),
            "wait_time_ns": self.wait_time_ns,
            "wait_time_us": round(self.wait_time_ns / 1_000, 2),
            "erisc_time_ns": self.erisc_time_ns,
            "erisc_time_us": round(self.erisc_time_ns / 1_000, 2),
            "io_overhead_percent": round(self.io_overhead_percent, 2),
            "is_io_bound": self.is_io_bound,
        }


class MultiCQAnalyzer:
    """Analyzer for multi-CQ (command queue) usage in TTNN operations."""

    def __init__(self, operations: list[OperationPerf]):
        """Initialize with operation performance data.

        Args:
            operations: List of OperationPerf objects from performance CSV.
        """
        self.operations = operations

    def get_summary(self) -> MultiCQSummary:
        """Calculate multi-CQ analysis summary.

        Returns:
            MultiCQSummary with analysis results.
        """
        if not self.operations:
            return MultiCQSummary(
                total_operations=0,
                total_device_time_ns=0,
                total_io_time_ns=0,
                total_dispatch_cq_time_ns=0,
                total_wait_time_ns=0,
                total_erisc_time_ns=0,
                total_compute_time_ns=0,
                io_overhead_percent=0,
                is_io_bound=False,
                multi_cq_recommended=False,
                io_bound_operations=0,
                recommendations=["No operations found in performance data"],
            )

        # Calculate totals
        total_device_time = sum(op.execution_time_ns for op in self.operations)
        total_dispatch_cq = sum(op.dispatch_cq_cmd_time_ns for op in self.operations)
        total_wait = sum(op.dispatch_wait_time_ns for op in self.operations)
        total_erisc = sum(op.erisc_kernel_duration_ns for op in self.operations)

        # Total I/O time is sum of dispatch, wait, and erisc times
        total_io_time = total_dispatch_cq + total_wait + total_erisc

        # Compute time is device time minus erisc time (erisc is data transfer)
        total_compute_time = max(0, total_device_time - total_erisc)

        # Calculate I/O overhead percentage
        total_time = total_device_time + total_dispatch_cq + total_wait
        if total_time > 0:
            io_overhead_percent = (total_io_time / total_time) * 100
        else:
            io_overhead_percent = 0

        # Count I/O-bound operations
        io_bound_ops = 0
        for op in self.operations:
            op_io_time = (
                op.dispatch_cq_cmd_time_ns
                + op.dispatch_wait_time_ns
                + op.erisc_kernel_duration_ns
            )
            op_total = op.execution_time_ns + op.dispatch_cq_cmd_time_ns + op.dispatch_wait_time_ns
            if op_total > 0 and (op_io_time / op_total) * 100 > IO_BOTTLENECK_THRESHOLD:
                io_bound_ops += 1

        # Determine recommendations
        is_io_bound = io_overhead_percent > IO_BOTTLENECK_THRESHOLD
        multi_cq_recommended = io_overhead_percent > MULTI_CQ_RECOMMENDED_THRESHOLD

        recommendations = self._generate_recommendations(
            io_overhead_percent=io_overhead_percent,
            is_io_bound=is_io_bound,
            multi_cq_recommended=multi_cq_recommended,
            total_dispatch_cq=total_dispatch_cq,
            total_wait=total_wait,
            total_erisc=total_erisc,
            io_bound_ops=io_bound_ops,
            total_ops=len(self.operations),
        )

        return MultiCQSummary(
            total_operations=len(self.operations),
            total_device_time_ns=total_device_time,
            total_io_time_ns=total_io_time,
            total_dispatch_cq_time_ns=total_dispatch_cq,
            total_wait_time_ns=total_wait,
            total_erisc_time_ns=total_erisc,
            total_compute_time_ns=total_compute_time,
            io_overhead_percent=io_overhead_percent,
            is_io_bound=is_io_bound,
            multi_cq_recommended=multi_cq_recommended,
            io_bound_operations=io_bound_ops,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        io_overhead_percent: float,
        is_io_bound: bool,
        multi_cq_recommended: bool,
        total_dispatch_cq: float,
        total_wait: float,
        total_erisc: float,
        io_bound_ops: int,
        total_ops: int,
    ) -> list[str]:
        """Generate optimization recommendations based on analysis.

        Args:
            io_overhead_percent: I/O overhead percentage.
            is_io_bound: Whether model is I/O-bound.
            multi_cq_recommended: Whether 2CQ is recommended.
            total_dispatch_cq: Total dispatch CQ time in ns.
            total_wait: Total wait time in ns.
            total_erisc: Total ERISC time in ns.
            io_bound_ops: Count of I/O-bound operations.
            total_ops: Total operation count.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if is_io_bound:
            recommendations.append(
                f"Model is I/O-BOUND ({io_overhead_percent:.1f}% I/O overhead): "
                "Device compute is waiting for data transfers"
            )

        if multi_cq_recommended:
            recommendations.append(
                "2CQ RECOMMENDED: Enable 2 command queues to overlap I/O with compute"
            )
            recommendations.append(
                "With 2CQ, one queue handles compute while the other handles data transfers"
            )

        # Identify the dominant I/O component
        total_io = total_dispatch_cq + total_wait + total_erisc
        if total_io > 0:
            if total_dispatch_cq / total_io > IO_DOMINANCE_THRESHOLD:
                recommendations.append(
                    f"Dispatch CQ time dominates I/O ({total_dispatch_cq/total_io*100:.0f}%): "
                    "Consider batching operations to reduce command overhead"
                )
            elif total_wait / total_io > IO_DOMINANCE_THRESHOLD:
                recommendations.append(
                    f"Wait time dominates I/O ({total_wait/total_io*100:.0f}%): "
                    "Consider async execution or pipelining"
                )
            elif total_erisc / total_io > IO_DOMINANCE_THRESHOLD:
                recommendations.append(
                    f"ERISC (data transfer) dominates I/O ({total_erisc/total_io*100:.0f}%): "
                    "Consider optimizing data placement or using sharding"
                )

        if io_bound_ops > 0:
            pct = (io_bound_ops / total_ops * 100) if total_ops > 0 else 0
            recommendations.append(
                f"{io_bound_ops} operations ({pct:.1f}%) are I/O-bound: "
                "Focus optimization on these operations"
            )

        if not multi_cq_recommended and not is_io_bound:
            recommendations.append(
                f"Model is COMPUTE-BOUND ({100-io_overhead_percent:.1f}% compute): "
                "Focus on kernel optimization rather than I/O overlap"
            )

        if not recommendations:
            recommendations.append("I/O overhead is within acceptable range")

        return recommendations

    def get_io_bound_operations(self, limit: int = 20) -> list[OperationIOAnalysis]:
        """Get operations with highest I/O overhead.

        Args:
            limit: Maximum number of operations to return.

        Returns:
            List of OperationIOAnalysis sorted by I/O overhead descending.
        """
        if not self.operations:
            return []

        analyses = []
        for op in self.operations:
            io_time = (
                op.dispatch_cq_cmd_time_ns
                + op.dispatch_wait_time_ns
                + op.erisc_kernel_duration_ns
            )
            total_time = op.execution_time_ns + op.dispatch_cq_cmd_time_ns + op.dispatch_wait_time_ns
            io_overhead = (io_time / total_time * 100) if total_time > 0 else 0
            is_io_bound = io_overhead > IO_BOTTLENECK_THRESHOLD

            analyses.append(
                OperationIOAnalysis(
                    op_code=op.op_code,
                    op_name=op.op_name,
                    device_time_ns=op.execution_time_ns,
                    dispatch_time_ns=op.dispatch_cq_cmd_time_ns,
                    wait_time_ns=op.dispatch_wait_time_ns,
                    erisc_time_ns=op.erisc_kernel_duration_ns,
                    io_overhead_percent=io_overhead,
                    is_io_bound=is_io_bound,
                )
            )

        # Sort by I/O overhead descending
        analyses.sort(key=lambda x: x.io_overhead_percent, reverse=True)
        return analyses[:limit]

    def get_io_distribution(self) -> dict[str, int]:
        """Get distribution of operations by I/O overhead level.

        Returns:
            Dictionary with I/O overhead ranges and counts.
        """
        if not self.operations:
            return {}

        distribution = {
            "0-10%": 0,
            "10-20%": 0,
            "20-30%": 0,
            "30-50%": 0,
            "50%+": 0,
        }

        for op in self.operations:
            io_time = (
                op.dispatch_cq_cmd_time_ns
                + op.dispatch_wait_time_ns
                + op.erisc_kernel_duration_ns
            )
            total_time = op.execution_time_ns + op.dispatch_cq_cmd_time_ns + op.dispatch_wait_time_ns
            if total_time == 0:
                continue
            io_overhead = (io_time / total_time) * 100

            if io_overhead < 10:
                distribution["0-10%"] += 1
            elif io_overhead < 20:
                distribution["10-20%"] += 1
            elif io_overhead < 30:
                distribution["20-30%"] += 1
            elif io_overhead < 50:
                distribution["30-50%"] += 1
            else:
                distribution["50%+"] += 1

        return distribution
