"""Tests for host overhead analysis module."""

import pytest

from ttnn_vis_cli.data.host_overhead_analysis import (
    HOST_BOUND_THRESHOLD,
    METAL_TRACE_RECOMMENDED_THRESHOLD,
    HostOverheadAnalyzer,
    HostOverheadSummary,
    OperationOverhead,
)
from ttnn_vis_cli.data.models import OperationPerf


def create_operation_perf(
    op_code: str = "ttnn::add",
    op_name: str = "add",
    execution_time_ns: float = 1000,
    op_to_op_gap_ns: float = 100,
    core_count: int = 64,
) -> OperationPerf:
    """Create a test OperationPerf object."""
    return OperationPerf(
        op_code=op_code,
        op_name=op_name,
        device_id=0,
        core_count=core_count,
        parallelization_strategy="",
        execution_time_ns=execution_time_ns,
        host_time_ns=0,
        math_utilization=0,
        dram_read_bw=0,
        dram_write_bw=0,
        l1_read_bw=0,
        l1_write_bw=0,
        op_to_op_gap_ns=op_to_op_gap_ns,
    )


class TestHostOverheadAnalyzer:
    """Tests for HostOverheadAnalyzer class."""

    def test_empty_operations(self):
        """Test with no operations."""
        analyzer = HostOverheadAnalyzer([])
        summary = analyzer.get_summary()

        assert summary.total_device_time_ns == 0
        assert summary.total_op_to_op_gap_ns == 0
        assert summary.operation_count == 0
        assert summary.is_host_bound is False
        assert summary.metal_trace_recommended is False

    def test_device_bound_model(self):
        """Test model that is device-bound (low host overhead)."""
        operations = [
            create_operation_perf(execution_time_ns=10000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=10000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=10000, op_to_op_gap_ns=100),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        # Total: 30000 device, 300 gap = 30300 e2e
        # Overhead: 300/30300 = ~1%
        assert summary.total_device_time_ns == 30000
        assert summary.total_op_to_op_gap_ns == 300
        assert summary.host_overhead_percent < 5
        assert summary.is_host_bound is False
        assert summary.metal_trace_recommended is False

    def test_host_bound_model(self):
        """Test model that is host-bound (high host overhead)."""
        operations = [
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=5000),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=5000),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=5000),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        # Total: 3000 device, 15000 gap = 18000 e2e
        # Overhead: 15000/18000 = ~83%
        assert summary.total_device_time_ns == 3000
        assert summary.total_op_to_op_gap_ns == 15000
        assert summary.host_overhead_percent > HOST_BOUND_THRESHOLD
        assert summary.is_host_bound is True
        assert summary.metal_trace_recommended is True

    def test_metal_trace_threshold(self):
        """Test Metal Trace recommendation threshold."""
        # Create operations with ~25% overhead (above recommendation threshold)
        operations = [
            create_operation_perf(execution_time_ns=3000, op_to_op_gap_ns=1000),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        # 1000/4000 = 25%
        assert summary.host_overhead_percent > METAL_TRACE_RECOMMENDED_THRESHOLD
        assert summary.metal_trace_recommended is True

    def test_get_top_overhead_operations(self):
        """Test getting operations sorted by overhead."""
        operations = [
            create_operation_perf(op_code="op1", execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(op_code="op2", execution_time_ns=1000, op_to_op_gap_ns=500),
            create_operation_perf(op_code="op3", execution_time_ns=1000, op_to_op_gap_ns=300),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        top_ops = analyzer.get_top_overhead_operations(limit=2)

        assert len(top_ops) == 2
        assert top_ops[0].op_code == "op2"  # Highest gap
        assert top_ops[1].op_code == "op3"  # Second highest
        assert top_ops[0].op_to_op_gap_ns == 500
        assert top_ops[1].op_to_op_gap_ns == 300

    def test_get_overhead_distribution(self):
        """Test overhead distribution calculation."""
        operations = [
            # 0-10% overhead
            create_operation_perf(execution_time_ns=10000, op_to_op_gap_ns=100),
            # 10-20% overhead
            create_operation_perf(execution_time_ns=8000, op_to_op_gap_ns=1500),
            # 30-50% overhead
            create_operation_perf(execution_time_ns=6000, op_to_op_gap_ns=4000),
            # 50%+ overhead
            create_operation_perf(execution_time_ns=2000, op_to_op_gap_ns=8000),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        distribution = analyzer.get_overhead_distribution()

        assert distribution["0-10%"] == 1
        assert distribution["10-20%"] == 1
        assert distribution["30-50%"] == 1
        assert distribution["50%+"] == 1

    def test_summary_statistics(self):
        """Test summary statistics calculations."""
        operations = [
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=2000, op_to_op_gap_ns=200),
            create_operation_perf(execution_time_ns=3000, op_to_op_gap_ns=300),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.operation_count == 3
        assert summary.avg_op_to_op_gap_ns == 200  # (100+200+300)/3
        assert summary.max_op_to_op_gap_ns == 300


class TestHostOverheadSummary:
    """Tests for HostOverheadSummary dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        summary = HostOverheadSummary(
            total_device_time_ns=1000000,
            total_op_to_op_gap_ns=200000,
            total_e2e_time_ns=1200000,
            host_overhead_percent=16.67,
            device_utilization_percent=83.33,
            operation_count=10,
            avg_op_to_op_gap_ns=20000,
            max_op_to_op_gap_ns=50000,
            is_host_bound=False,
            metal_trace_recommended=False,
            recommendations=["Test recommendation"],
        )

        result = summary.to_dict()

        assert result["total_device_time_ns"] == 1000000
        assert result["total_device_time_ms"] == 1.0
        assert result["host_overhead_percent"] == 16.67
        assert result["is_host_bound"] is False
        assert result["recommendations"] == ["Test recommendation"]


class TestOperationOverhead:
    """Tests for OperationOverhead dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        overhead = OperationOverhead(
            op_code="ttnn::matmul",
            op_name="matmul",
            device_time_ns=5000,
            op_to_op_gap_ns=1000,
            overhead_percent=16.67,
            core_count=64,
        )

        result = overhead.to_dict()

        assert result["op_code"] == "ttnn::matmul"
        assert result["device_time_ns"] == 5000
        assert result["device_time_us"] == 5.0
        assert result["op_to_op_gap_us"] == 1.0
        assert result["overhead_percent"] == 16.67


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_host_bound_recommendation(self):
        """Test recommendation for host-bound model."""
        operations = [
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=10000),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("HOST-BOUND" in rec for rec in summary.recommendations)
        assert any("METAL TRACE" in rec for rec in summary.recommendations)

    def test_device_bound_recommendation(self):
        """Test recommendation for device-bound model."""
        operations = [
            create_operation_perf(execution_time_ns=10000, op_to_op_gap_ns=100),
        ]
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("DEVICE-BOUND" in rec for rec in summary.recommendations)

    def test_large_gap_variance_recommendation(self):
        """Test recommendation for large gap variance."""
        # Need max_gap > avg_gap * 3 and max_gap > 10000ns
        # With gaps [100, 100, 100000], avg=33400, max=100000
        # 100000 < 33400*3=100200, so use smaller gaps for first two
        operations = [
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=100),
            create_operation_perf(execution_time_ns=1000, op_to_op_gap_ns=50000),  # Large gap
        ]
        # avg = (100*4 + 50000) / 5 = 10080, max = 50000
        # 50000 > 10080 * 3 = 30240 ✓, 50000 > 10000 ✓
        analyzer = HostOverheadAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("variance" in rec.lower() for rec in summary.recommendations)
