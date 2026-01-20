"""Host overhead analysis logic for TTNN profiling data."""

from dataclasses import dataclass

from .models import OperationPerf

# Thresholds for recommendations
HOST_BOUND_THRESHOLD = 30  # Consider host-bound if overhead > 30%
METAL_TRACE_RECOMMENDED_THRESHOLD = 20  # Recommend Metal Trace if overhead > 20%
LARGE_GAP_VARIANCE_MULTIPLIER = 3  # Flag variance if max > avg * this
LARGE_GAP_MIN_THRESHOLD_NS = 10000  # Minimum gap (10us) to flag variance


@dataclass
class HostOverheadSummary:
    """Summary of host overhead analysis."""

    total_device_time_ns: float
    total_op_to_op_gap_ns: float
    total_e2e_time_ns: float
    host_overhead_percent: float
    device_utilization_percent: float
    operation_count: int
    avg_op_to_op_gap_ns: float
    max_op_to_op_gap_ns: float
    is_host_bound: bool
    metal_trace_recommended: bool
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_device_time_ns": self.total_device_time_ns,
            "total_device_time_ms": round(self.total_device_time_ns / 1_000_000, 3),
            "total_op_to_op_gap_ns": self.total_op_to_op_gap_ns,
            "total_op_to_op_gap_ms": round(self.total_op_to_op_gap_ns / 1_000_000, 3),
            "total_e2e_time_ns": self.total_e2e_time_ns,
            "total_e2e_time_ms": round(self.total_e2e_time_ns / 1_000_000, 3),
            "host_overhead_percent": round(self.host_overhead_percent, 2),
            "device_utilization_percent": round(self.device_utilization_percent, 2),
            "operation_count": self.operation_count,
            "avg_op_to_op_gap_ns": round(self.avg_op_to_op_gap_ns, 2),
            "max_op_to_op_gap_ns": self.max_op_to_op_gap_ns,
            "is_host_bound": self.is_host_bound,
            "metal_trace_recommended": self.metal_trace_recommended,
            "recommendations": self.recommendations,
        }


@dataclass
class OperationOverhead:
    """Host overhead information for a single operation."""

    op_code: str
    op_name: str
    device_time_ns: float
    op_to_op_gap_ns: float
    overhead_percent: float
    core_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "op_code": self.op_code,
            "op_name": self.op_name,
            "device_time_ns": self.device_time_ns,
            "device_time_us": round(self.device_time_ns / 1_000, 2),
            "op_to_op_gap_ns": self.op_to_op_gap_ns,
            "op_to_op_gap_us": round(self.op_to_op_gap_ns / 1_000, 2),
            "overhead_percent": round(self.overhead_percent, 2),
            "core_count": self.core_count,
        }


class HostOverheadAnalyzer:
    """Analyzer for host overhead in TTNN operations."""

    def __init__(self, operations: list[OperationPerf]):
        """Initialize with operation performance data.

        Args:
            operations: List of OperationPerf objects from performance CSV.
        """
        self.operations = operations

    def get_summary(self) -> HostOverheadSummary:
        """Calculate host overhead summary.

        Returns:
            HostOverheadSummary with analysis results.
        """
        if not self.operations:
            return HostOverheadSummary(
                total_device_time_ns=0,
                total_op_to_op_gap_ns=0,
                total_e2e_time_ns=0,
                host_overhead_percent=0,
                device_utilization_percent=0,
                operation_count=0,
                avg_op_to_op_gap_ns=0,
                max_op_to_op_gap_ns=0,
                is_host_bound=False,
                metal_trace_recommended=False,
                recommendations=["No operations found in performance data"],
            )

        # Calculate totals
        total_device_time = sum(op.execution_time_ns for op in self.operations)
        total_op_to_op_gap = sum(op.op_to_op_gap_ns for op in self.operations)
        total_e2e_time = total_device_time + total_op_to_op_gap

        # Calculate percentages
        if total_e2e_time > 0:
            host_overhead_percent = (total_op_to_op_gap / total_e2e_time) * 100
            device_utilization_percent = (total_device_time / total_e2e_time) * 100
        else:
            host_overhead_percent = 0
            device_utilization_percent = 0

        # Calculate averages and max
        op_count = len(self.operations)
        avg_gap = total_op_to_op_gap / op_count if op_count > 0 else 0
        max_gap = max(op.op_to_op_gap_ns for op in self.operations) if self.operations else 0

        # Determine if host-bound
        is_host_bound = host_overhead_percent > HOST_BOUND_THRESHOLD
        metal_trace_recommended = host_overhead_percent > METAL_TRACE_RECOMMENDED_THRESHOLD

        # Generate recommendations
        recommendations = self._generate_recommendations(
            host_overhead_percent=host_overhead_percent,
            is_host_bound=is_host_bound,
            metal_trace_recommended=metal_trace_recommended,
            avg_gap=avg_gap,
            max_gap=max_gap,
        )

        return HostOverheadSummary(
            total_device_time_ns=total_device_time,
            total_op_to_op_gap_ns=total_op_to_op_gap,
            total_e2e_time_ns=total_e2e_time,
            host_overhead_percent=host_overhead_percent,
            device_utilization_percent=device_utilization_percent,
            operation_count=op_count,
            avg_op_to_op_gap_ns=avg_gap,
            max_op_to_op_gap_ns=max_gap,
            is_host_bound=is_host_bound,
            metal_trace_recommended=metal_trace_recommended,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        host_overhead_percent: float,
        is_host_bound: bool,
        metal_trace_recommended: bool,
        avg_gap: float,
        max_gap: float,
    ) -> list[str]:
        """Generate optimization recommendations based on analysis.

        Args:
            host_overhead_percent: Host overhead percentage.
            is_host_bound: Whether model is host-bound.
            metal_trace_recommended: Whether Metal Trace is recommended.
            avg_gap: Average op-to-op gap in ns.
            max_gap: Maximum op-to-op gap in ns.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if is_host_bound:
            recommendations.append(
                f"Model is HOST-BOUND ({host_overhead_percent:.1f}% overhead): "
                "Device is waiting for host dispatch"
            )

        if metal_trace_recommended:
            recommendations.append(
                "METAL TRACE RECOMMENDED: Capture and replay operations to eliminate host overhead"
            )
            recommendations.append(
                "Prerequisites: All tensor shapes must be static, same operations run repeatedly"
            )

        if max_gap > avg_gap * LARGE_GAP_VARIANCE_MULTIPLIER and max_gap > LARGE_GAP_MIN_THRESHOLD_NS:
            recommendations.append(
                f"Large gap variance detected (max: {max_gap/1000:.1f}us, avg: {avg_gap/1000:.1f}us): "
                "Investigate specific operations with high gaps"
            )

        if host_overhead_percent < METAL_TRACE_RECOMMENDED_THRESHOLD:
            recommendations.append(
                f"Model is DEVICE-BOUND ({100-host_overhead_percent:.1f}% device utilization): "
                "Focus on kernel optimization rather than host overhead"
            )

        if not recommendations:
            recommendations.append("Host overhead is within acceptable range")

        return recommendations

    def get_top_overhead_operations(
        self, limit: int = 20
    ) -> list[OperationOverhead]:
        """Get operations with highest op-to-op gaps.

        Args:
            limit: Maximum number of operations to return.

        Returns:
            List of OperationOverhead sorted by gap descending.
        """
        if not self.operations:
            return []

        overheads = []
        for op in self.operations:
            total_time = op.execution_time_ns + op.op_to_op_gap_ns
            overhead_percent = (
                (op.op_to_op_gap_ns / total_time * 100) if total_time > 0 else 0
            )
            overheads.append(
                OperationOverhead(
                    op_code=op.op_code,
                    op_name=op.op_name,
                    device_time_ns=op.execution_time_ns,
                    op_to_op_gap_ns=op.op_to_op_gap_ns,
                    overhead_percent=overhead_percent,
                    core_count=op.core_count,
                )
            )

        # Sort by op_to_op_gap descending
        overheads.sort(key=lambda x: x.op_to_op_gap_ns, reverse=True)
        return overheads[:limit]

    def get_overhead_distribution(self) -> dict[str, int]:
        """Get distribution of operations by overhead level.

        Returns:
            Dictionary with overhead ranges and counts.
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
            total_time = op.execution_time_ns + op.op_to_op_gap_ns
            if total_time == 0:
                continue
            overhead_percent = (op.op_to_op_gap_ns / total_time) * 100

            if overhead_percent < 10:
                distribution["0-10%"] += 1
            elif overhead_percent < 20:
                distribution["10-20%"] += 1
            elif overhead_percent < 30:
                distribution["20-30%"] += 1
            elif overhead_percent < 50:
                distribution["30-50%"] += 1
            else:
                distribution["50%+"] += 1

        return distribution
